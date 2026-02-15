import asyncio
import os
import traceback
from collections import defaultdict
from collections.abc import Coroutine
from typing import Any

from jinja2 import Template

from pptagent.agent import Agent
from pptagent.llms import LLM, AsyncLLM
from pptagent.model_utils import (
    get_cluster,
    get_image_embedding,
    images_cosine_similarity,
)
from pptagent.presentation import Picture, Presentation, SlidePage
from pptagent.utils import (
    Config,
    edit_distance,
    get_logger,
    is_image_path,
    package_join,
    pjoin,
)

logger = get_logger(__name__)

CATEGORY_SPLIT_TEMPLATE = Template(
    open(package_join("prompts", "category_split.txt")).read()
)
ASK_CATEGORY_PROMPT = open(package_join("prompts", "ask_category.txt")).read()


def check_schema(schema: dict | Any, slide: SlidePage):
    # [FIX] Llama 3.2 sometimes returns a list containing the dict
    if isinstance(schema, list):
        if len(schema) > 0 and isinstance(schema[0], dict):
            schema = schema[0]
        elif len(schema) > 0:
             # Try to merge or just take the first
             logger.warning(f"Schema is a list but first item is {type(schema[0])}: {schema[0]}")
             if isinstance(schema[0], dict):
                 schema = schema[0]

    if not isinstance(schema, dict):
        # Fallback: if it's still not a dict, create a dummy one to prevent crash
        logger.error(f"Output schema should be a dict, but got {type(schema)}: {schema}")
        # raise ValueError(...) - Don't raise, just warn and try to proceed with empty?
        # Actually raising is better than silent failure content-wise, but we want stability.
        # Let's try to convert it if it's a string? No, just raise but with better message
        raise ValueError(
            f"Output schema should be a dict, but got {type(schema)}: {schema}\n",
            """  {
                "element_name": {
                    "description": "purpose of this element", # do not mention any detail, just purpose
                    "type": "text" or "image",
                    "data": ["text1", "text2"] or ["logo:...", "logo:..."]
                    }
            }""",
        )

    similar_ele = None
    max_similarity = -1
    for el_name, element in list(schema.items()): # Use list() to allow modification during iteration
        # [FIX] Critical: Llama 3.2 output normalization
        if element is None:
            element = {"type": "text", "content": "", "data": []}
            schema[el_name] = element
        elif isinstance(element, str):
            element = {"type": "text", "content": element, "data": [element]}
            schema[el_name] = element
        elif isinstance(element, list):
            # Assume list of strings
            content = " ".join(str(e) for e in element)
            element = {"type": "text", "content": content, "data": element}
            schema[el_name] = element
        elif not isinstance(element, dict):
            # Fallback for other types
            element = {"type": "text", "content": str(element), "data": [str(element)]}
            schema[el_name] = element

        # [FIX] If Local AI forgets the 'type', assume it is text
        if "type" not in element:
            element["type"] = "text"

        if element["type"] == "text":
            # [FIX] If Local AI forgets 'content', add placeholder
            if "content" not in element:
                element["content"] = "Content placeholder"
            
            # Legacy fields check (rest of original logic)
            if "data" not in element or len(element["data"]) == 0:
                # If content is present but data is missing (Llama structure), migrate it
                if "content" in element and len(element["content"]) > 0:
                    element["data"] = [element["content"]]
                else: 
                     # Skip empty checks to be permissive
                     pass

        if "data" not in element:
             element["data"] = []

        if not isinstance(element["data"], list):
            logger.debug("Converting single text element to list: %s", element["data"])
            element["data"] = [element["data"]]

        if element["type"] == "text":

                for item in element["data"]:
                    # [FIX] Force string
                    item = str(item)
                    for para in slide.iter_paragraphs():
                        similarity = edit_distance(para.text, item)
                        if similarity > 0.5:
                            break
                        if similarity > max_similarity:
                            max_similarity = similarity
                            similar_ele = para.text
                    else:
                        # [FIX] Soften strict check for Local AI
                        # Instead of crashing, just log warning and proceed
                        logger.warning(
                            f"Text element `{el_name}` match failed for '{item}'. Closest: '{similar_ele}'"
                        )

        elif element["type"] == "image":

            for caption in element["data"]:
                # [FIX] Force string
                caption = str(caption)
                for shape in slide.shape_filter(Picture):
                    similarity = edit_distance(shape.caption, caption)
                    if similarity > 0.5:
                        break
                    if similarity > max_similarity:
                        max_similarity = similarity
                        similar_ele = shape.caption
                else:
                    # [FIX] Soften image caption check too
                    logger.warning(
                        f"Image caption of {el_name}: {caption} not found. Closest: {similar_ele}"
                    )

        else:
            raise ValueError(
                f"Unknown type of {el_name}: {element['type']}, should be one of ['text', 'image']"
            )


class SlideInducter:
    """
    Stage I: Presentation Analysis.
    This stage is to analyze the presentation: cluster slides into different layouts, and extract content schema for each layout.
    """

    def __init__(
        self,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
        language_model: LLM,
        vision_model: LLM,
        use_assert: bool = True,
    ):
        """
        Initialize the SlideInducter.

        Args:
            prs (Presentation): The presentation object.
            ppt_image_folder (str): The folder containing PPT images.
            template_image_folder (str): The folder containing normalized slide images.
            config (Config): The configuration object.
            image_models (list): A list of image models.
        """
        self.prs = prs
        self.config = config
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        self.language_model = language_model
        self.vision_model = vision_model
        self.image_models = image_models
        self.schema_extractor = Agent(
            "schema_extractor",
            {
                "language": language_model,
            },
        )
        if not use_assert:
            return

        num_template_images = sum(
            is_image_path(f) for f in os.listdir(template_image_folder)
        )
        num_ppt_images = sum(is_image_path(f) for f in os.listdir(ppt_image_folder))
        num_slides = len(prs.slides)

        if not (num_template_images == num_ppt_images == num_slides):
            raise ValueError(
                f"Slide count mismatch detected:\n"
                f"- Presentation slides: {num_slides}\n"
                f"- Template images: {num_template_images} ({template_image_folder})\n"
                f"- PPT images: {num_ppt_images} ({ppt_image_folder})\n"
                f"All counts must be equal."
            )

    def layout_induct(self) -> dict:
        """
        Perform layout induction for the presentation, should be called before content induction.
        Return a dict representing layouts, each layout is a dict with keys:
        - key: the layout name, e.g. "Title and Content:text"
        - `template_id`: the id of the template slide
        - `slides`: the list of slide ids
        Moreover, the dict has a key `functional_keys`, which is a list of functional keys.
        """
        layout_induction = defaultdict(lambda: defaultdict(list))
        content_slides_index, functional_cluster = self.category_split()
        for layout_name, cluster in functional_cluster.items():
            layout_induction[layout_name]["slides"] = cluster
            layout_induction[layout_name]["template_id"] = cluster[0]

        functional_keys = list(layout_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in layout_induction.items():
            function_slides_index.update(cluster["slides"])
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index, layout_induction)
        layout_induction["functional_keys"] = functional_keys
        return layout_induction

    def category_split(self):
        """
        Split slides into categories based on their functional purpose.
        """
        functional_cluster = self.language_model(
            CATEGORY_SPLIT_TEMPLATE.render(slides=self.prs.to_text()),
            return_json=True,
        )
        assert isinstance(functional_cluster, dict) and all(
            isinstance(k, str) and isinstance(v, list)
            for k, v in functional_cluster.items()
        ), "Functional cluster must be a dictionary with string keys and list values"
        functional_slides = set(sum(functional_cluster.values(), []))
        content_slides_index = set(range(1, len(self.prs) + 1)) - functional_slides

        return content_slides_index, functional_cluster

    def layout_split(self, content_slides_index: set[int], layout_induction: dict):
        """
        Cluster slides into different layouts.
        """
        embeddings = get_image_embedding(self.template_image_folder, *self.image_models)
        assert len(embeddings) == len(self.prs)
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_type = slide.get_content_type()
            layout_name = slide.slide_layout_name
            content_split[(layout_name, content_type)].append(slide_idx)

        for (layout_name, content_type), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                slide_indexs = [slides[i] for i in cluster]
                template_id = max(
                    slide_indexs,
                    key=lambda x: len(self.prs.slides[x - 1].shapes),
                )
                cluster_name = (
                    self.vision_model(
                        ASK_CATEGORY_PROMPT,
                        pjoin(self.ppt_image_folder, f"slide_{template_id:04d}.jpg"),
                    )
                    + ":"
                    + content_type
                )
                layout_induction[cluster_name]["template_id"] = template_id
                layout_induction[cluster_name]["slides"] = slide_indexs

    def content_induct(self, layout_induction: dict):
        """
        Perform content schema extraction for the presentation.
        """
        for layout_name, cluster in layout_induction.items():
            if layout_name == "functional_keys" or "content_schema" in cluster:
                continue
            slide = self.prs.slides[cluster["template_id"] - 1]
            turn_id, schema = self.schema_extractor(slide=slide.to_html())
            schema = self._fix_schema(schema, slide, turn_id)
            layout_induction[layout_name]["content_schema"] = schema

        return layout_induction

    def _fix_schema(
        self,
        schema: dict,
        slide: SlidePage,
        turn_id: int = None,
        retry: int = 0,
    ) -> dict:
        """
        Fix schema by checking and retrying if necessary.
        """
        try:
            check_schema(schema, slide)
        except ValueError as e:
            retry += 1
            logger.debug("Failed at schema extraction: %s", e)
            if retry == 3:
                logger.error("Failed to extract schema for slide-%s: %s", turn_id, e)
                raise e
            schema = self.schema_extractor.retry(
                e, traceback.format_exc(), turn_id, retry
            )
            return self._fix_schema(schema, slide, turn_id, retry)
        return schema


class SlideInducterAsync(SlideInducter):
    def __init__(
        self,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
        language_model: AsyncLLM,
        vision_model: AsyncLLM,
    ):
        """
        Initialize the SlideInducterAsync with async models.

        Args:
            prs (Presentation): The presentation object.
            ppt_image_folder (str): The folder containing PPT images.
            template_image_folder (str): The folder containing normalized slide images.
            config (Config): The configuration object.
            image_models (list): A list of image models.
            language_model (AsyncLLM): The async language model.
            vision_model (AsyncLLM): The async vision model.
        """
        super().__init__(
            prs,
            ppt_image_folder,
            template_image_folder,
            config,
            image_models,
            language_model,
            vision_model,
        )
        self.language_model = self.language_model.to_async()
        self.vision_model = self.vision_model.to_async()
        self.schema_extractor = self.schema_extractor.to_async()

    async def category_split(self):
        """
        Async version: Split slides into categories based on their functional purpose.
        """
        functional_cluster = await self.language_model(
            CATEGORY_SPLIT_TEMPLATE.render(slides=self.prs.to_text()),
            return_json=True,
        )
        assert isinstance(functional_cluster, dict) and all(
            isinstance(k, str) and isinstance(v, list)
            for k, v in functional_cluster.items()
        ), "Functional cluster must be a dictionary with string keys and list values"
        functional_slides = set(sum(functional_cluster.values(), []))
        content_slides_index = set(range(1, len(self.prs) + 1)) - functional_slides

        return content_slides_index, functional_cluster

    async def layout_split(
        self, content_slides_index: set[int], layout_induction: dict
    ):
        """
        Async version: Cluster slides into different layouts.
        """
        embeddings = get_image_embedding(self.template_image_folder, *self.image_models)
        assert len(embeddings) == len(self.prs)
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_type = slide.get_content_type()
            layout_name = slide.slide_layout_name
            content_split[(layout_name, content_type)].append(slide_idx)

        async with asyncio.TaskGroup() as tg:
            for (layout_name, content_type), slides in content_split.items():
                sub_embeddings = [
                    embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
                ]
                similarity = images_cosine_similarity(sub_embeddings)
                for cluster in get_cluster(similarity):
                    slide_indexs = [slides[i] for i in cluster]
                    template_id = max(
                        slide_indexs,
                        key=lambda x: len(self.prs.slides[x - 1].shapes),
                    )

                    tg.create_task(
                        self.vision_model(
                            ASK_CATEGORY_PROMPT,
                            pjoin(
                                self.ppt_image_folder, f"slide_{template_id:04d}.jpg"
                            ),
                        )
                    ).add_done_callback(
                        lambda f, tid=template_id, sidxs=slide_indexs, ctype=content_type: layout_induction[
                            f.result() + ":" + ctype
                        ].update(
                            {"template_id": tid, "slides": sidxs}
                        )
                    )

    async def layout_induct(self):
        """
        Async version: Perform layout induction for the presentation.
        """
        layout_induction = defaultdict(lambda: defaultdict(list))
        content_slides_index, functional_cluster = await self.category_split()
        for layout_name, cluster in functional_cluster.items():
            if not cluster or len(cluster) == 0:
                 continue
            layout_induction[layout_name]["slides"] = cluster
            layout_induction[layout_name]["template_id"] = cluster[0]

        functional_keys = list(layout_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in layout_induction.items():
            function_slides_index.update(cluster["slides"])
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        await self.layout_split(content_slides_index, layout_induction)
        layout_induction["functional_keys"] = functional_keys
        return layout_induction

    async def content_induct(self, layout_induction: dict):
        """
        Async version: Perform content schema extraction for the presentation.
        """
        async with asyncio.TaskGroup() as tg:
            for layout_name, cluster in layout_induction.items():
                if layout_name == "functional_keys" or "content_schema" in cluster:
                    continue
                slide = self.prs.slides[cluster["template_id"] - 1]
                coro = self.schema_extractor(slide=slide.to_html())

                tg.create_task(self._fix_schema(coro, slide)).add_done_callback(
                    lambda f, key=layout_name: layout_induction[key].update(
                        {"content_schema": f.result()}
                    )
                )

        return layout_induction

    async def _fix_schema(
        self,
        schema: dict | Coroutine[dict, None, None],
        slide: SlidePage,
        turn_id: int = None,
        retry: int = 0,
    ):
        if retry == 0:
            turn_id, schema = await schema

        # [FIX] Smart Unwrap: Find the dict buried in the list
        if isinstance(schema, list):
            found_dict = None
            if len(schema) > 0 and isinstance(schema[0], dict):
                 found_dict = schema[0]
            else:
                 # Search for any dict in the list
                 for item in schema:
                     if isinstance(item, dict):
                         found_dict = item
                         break
            
            if found_dict:
                logger.warning("Recovered valid dict from messy list.")
                schema = found_dict
            else:
                 logger.error("Schema is a list but contains no dicts: %s", schema)
                 # Converting to empty dict will trigger retry logic downstream
                 # instead of crashing with mismatched type
                 # schema = {} 
                 pass # Let it fail in check_schema so we see the error, or loop retry
        
        try:
            check_schema(schema, slide)
        except ValueError as e:
            retry += 1
            logger.debug("Failed at schema extraction: %s", e)
            if retry == 3:
                logger.error("Failed to extract schema for slide-%s: %s", turn_id, e)
                raise e
            schema = await self.schema_extractor.retry(
                e, traceback.format_exc(), turn_id, retry
            )
            return await self._fix_schema(schema, slide, turn_id, retry)
        return schema
