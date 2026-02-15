import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from jinja2 import StrictUndefined, Template

from pptagent.llms import LLM, AsyncLLM
from pptagent.utils import get_logger, package_join, pbasename, pexists, pjoin

logger = get_logger(__name__)

LENGTHY_REWRITE_PROMPT = Template(
    open(package_join("prompts", "lengthy_rewrite.txt")).read(),
    undefined=StrictUndefined,
)


@dataclass
class Element:
    el_name: str
    content: list[str]
    description: str
    el_type: Literal["text", "image"]
    suggested_characters: int | None
    variable_length: tuple[int, int] | None
    variable_data: dict[str, list[str]] | None

    def get_schema(self):
        schema = f"Element: {self.el_name}\n"
        base_attrs = ["description", "el_type"]
        for attr in base_attrs:
            schema += f"\t{attr}: {getattr(self, attr)}\n"
        if self.el_type == "text":
            schema += f"\tsuggested_characters: {self.suggested_characters}\n"
        if self.variable_length is not None:
            schema += f"\tThe length of the element can vary between {self.variable_length[0]} and {self.variable_length[1]}\n"
        schema += f"\tThe default quantity of the element is {len(self.content)}\n"
        return schema

    @classmethod
    def from_dict(cls, el_name: str, data: dict):
        if not isinstance(data["data"], list):
            data["data"] = [data["data"]]
        if data["type"] == "text":
            suggested_characters = max(len(i) for i in data["data"])
        elif data["type"] == "image":
            suggested_characters = None
        return cls(
            el_name=el_name,
            # [FIX] Robust parsing for missing keys
            description=data.get("description", ""),
            el_type=data.get("type", "text"),
            content=data.get("data", []),
            variable_length=data.get("variableLength", None),
            variable_data=data.get("variableData", None),
            suggested_characters=suggested_characters,
        )


@dataclass
class Layout:
    title: str
    template_id: int
    slides: list[int]
    elements: list[Element]
    vary_mapping: dict[int, int] | None  # mapping for variable elements

    @classmethod
    def from_dict(cls, title: str, data: dict):
        elements = []
        for el_name, val in data["content_schema"].items():
            if isinstance(val, dict):
                elements.append(Element.from_dict(el_name, val))
            else:
                 # Fallback if the LLM sent a string instead of a dict
                 logger.warning(f"Element {el_name} has invalid schema: {val}. Using fallback.")
                 elements.append(Element.from_dict(el_name, {"description": str(val), "type": "text", "data": [], "content": []}))
        num_vary_elements = sum((el.variable_length is not None) for el in elements)
        if num_vary_elements > 1:
            raise ValueError("Only one variable element is allowed")
        return cls(
            title=title,
            template_id=data["template_id"],
            slides=data["slides"],
            elements=elements,
            vary_mapping=data.get("vary_mapping", None),
        )

    def get_slide_id(self, data: dict):
        for el in self.elements:
            if el.variable_length is not None:
                num_vary = len(data[el.el_name]["data"])
                if num_vary < el.variable_length[0]:
                    raise ValueError(
                        f"The length of {el.el_name}: {num_vary} is less than the minimum length: {el.variable_length[0]}"
                    )
                if num_vary > el.variable_length[1]:
                    raise ValueError(
                        f"The length of {el.el_name}: {num_vary} is greater than the maximum length: {el.variable_length[1]}"
                    )
                return self.vary_mapping[str(num_vary)]
        return self.template_id

    def get_old_data(self, editor_output: Optional[dict] = None):
        if editor_output is None:
            return {el.el_name: el.content for el in self.elements}
        old_data = {}
        for el in self.elements:
            if el.variable_length is not None:
                key = str(len(editor_output[el.el_name]["data"]))
                assert (
                    key in el.variable_data
                ), f"The length of element {el.el_name} varies between {el.variable_length[0]} and {el.variable_length[1]}, but got data of length {key} which is not supported"
                old_data[el.el_name] = el.variable_data[key]
            else:
                old_data[el.el_name] = el.content
        return old_data

    def validate(self, editor_output: dict, image_dir: str):
        # [FIX] Flexible Element Mapping
        # Create a map of normalized_key -> actual_key
        # e.g. "maintitle" -> "main title"
        supported_map = {
            el.el_name.replace(" ", "").replace("_", "").lower(): el.el_name
            for el in self.elements
        }

        # Create a clean output with correct keys
        clean_output = {}
        for el_name, el_data in editor_output.items():
            normalized_name = el_name.replace("_", "").replace(" ", "").lower()
            if normalized_name in supported_map:
                actual_name = supported_map[normalized_name]
                clean_output[actual_name] = el_data
            else:
                logger.warning(f"Skipping unknown element: {el_name}")

        # Update editor_output in-place with corrected keys
        editor_output.clear()
        
        # [FIX] Filter out fake paths
        final_clean = {}
        for el_name, el_data in clean_output.items():
             if el_data.get("type") == "image":
                 data_list = el_data.get("data", [])
                 if data_list and (data_list[0] == "/path/to/image" or not os.path.exists(data_list[0])):
                     logger.warning(f"Removing invalid image path: {data_list[0]}")
                     continue
             final_clean[el_name] = el_data

        editor_output.update(final_clean)

        for el_name, el_data in editor_output.items():
            assert (
                "data" in el_data
            ), """key `data` not found in output..."""
            
            assert (
                el_name in self
            ), f"Element {el_name} is not a valid element..."

            if self[el_name].el_type == "image":
                # Ensure data is list
                if not isinstance(el_data["data"], list):
                     el_data["data"] = [el_data["data"]]

                for i in range(len(el_data["data"])):
                    # Previous logic for image validation
                    path = el_data["data"][i]
                    if pexists(pjoin(image_dir, path)):
                        el_data["data"][i] = pjoin(image_dir, path)
                    elif not pexists(path):
                        basename = pbasename(path)
                        if pexists(pjoin(image_dir, basename)):
                             el_data["data"][i] = pjoin(image_dir, basename)
                        else:
                            # Soften image failure to warning
                            logger.warning(f"Image {path} not found. Ignoring.")
                            el_data["data"][i] = "" # Clear invalid path

    def validate_length(
        self, editor_output: dict, length_factor: float, language_model: LLM
    ):
        for el_name, el_data in editor_output.items():
            if self[el_name].el_type == "text":
                charater_counts = [len(i) for i in el_data["data"]]
                if (
                    max(charater_counts)
                    > self[el_name].suggested_characters * length_factor
                ):
                    el_data["data"] = language_model(
                        LENGTHY_REWRITE_PROMPT.render(
                            el_name=el_name,
                            content=el_data["data"],
                            suggested_characters=f"{self[el_name].suggested_characters} characters",
                        ),
                        return_json=True,
                    )
                    assert isinstance(
                        el_data["data"], list
                    ), f"Generated data is lengthy, expect {self[el_name].suggested_characters} characters, but got {len(el_data['data'])} characters for element {el_name}"

    async def validate_length_async(
        self, editor_output: dict, length_factor: float, language_model: AsyncLLM
    ):
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = {}
                for el_name, el_data in editor_output.items():
                    if self[el_name].el_type == "text":
                        charater_counts = [len(i) for i in el_data["data"]]
                        if (
                            max(charater_counts)
                            > self[el_name].suggested_characters * length_factor
                        ):
                            task = tg.create_task(
                                language_model(
                                    LENGTHY_REWRITE_PROMPT.render(
                                        el_name=el_name,
                                        content=el_data["data"],
                                        suggested_characters=f"{self[el_name].suggested_characters} characters",
                                    ),
                                    return_json=True,
                                )
                            )
                            tasks[el_name] = task

            for el_name, task in tasks.items():
                assert isinstance(
                    editor_output[el_name]["data"], list
                ), f"Generated data is lengthy, expect {self[el_name].suggested_characters} characters, but got {len(editor_output[el_name]['data'])} characters for element {el_name}"
                new_data = await task
                logger.debug(
                    f"Lengthy rewrite for {el_name}:\n From {editor_output[el_name]['data']}\n To {new_data}"
                )
                editor_output[el_name]["data"] = new_data
        
        except Exception as e:
            import traceback
            print("\n!!! LENGTH VALIDATION ERROR !!!")
            if hasattr(e, 'exceptions'):
                for i, exc in enumerate(e.exceptions):
                    print(f"Sub-exception {i+1}: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
            else:
                print(f"Error: {e}")
                traceback.print_exc()
            raise e

    @property
    def content_schema(self):
        return "\n".join([el.get_schema() for el in self.elements])

    def remove_item(self, item: str):
        for el in self.elements:
            if item in el.content:
                el.content.remove(item)
                if len(el.content) == 0:
                    self.elements.remove(el)
                return
        else:
            raise ValueError(f"Item {item} not found in layout {self.title}")

    def __contains__(self, key: str | int):
        if isinstance(key, int):
            return key in self.slides
        elif isinstance(key, str):
            for el in self.elements:
                if el.el_name == key:
                    return True
            return False
        raise ValueError(f"Invalid key type: {type(key)}, should be str or int")

    def __getitem__(self, key: str):
        for el in self.elements:
            if el.el_name == key:
                return el
        raise ValueError(f"Element {key} not found")

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)
