<template>
  <!-- Upload form -->
  <ElectricBorder class="upload-container" color="#7df9ff">
    <div class="upload-options">
      <!-- Row 1: Upload Buttons -->
      <div class="upload-buttons">
        <div class="upload-section">
          <label for="pptx-upload" class="upload-label">
            Upload PPT Template
            <span v-if="pptxFile" class="uploaded-symbol">✔️</span>
          </label>
          <input type="file" id="pptx-upload" @change="handleFileUpload($event, 'pptx')" accept=".pptx" />
        </div>
        <div class="upload-section">
          <label for="pdf-upload" class="upload-label">
            Upload File
            <span v-if="pdfFile" class="uploaded-symbol">✔️</span>
          </label>
          <input type="file" id="pdf-upload" @change="handleFileUpload($event, 'pdf')" accept=".pdf" />
        </div>
      </div>

      <!-- Row 2: URL Input (New Feature) -->
      <div class="selectors">
         <div class="pages-selection" style="width: 80%;">
            <input type="text" v-model="url" placeholder="OR Paste Website URL here..." class="url-input" />
         </div>
      </div>

      <!-- Row 3: Selectors & Settings -->
      <div class="selectors">
        <div class="pages-selection">
          <input type="text" v-model="topic" placeholder="Topic / Instructions (Optional)" class="topic-input" />
        </div>
        <div class="pages-selection">
          <select v-model="selectedPages">
            <option v-for="page in pagesOptions" :key="page" :value="page">{{ page }} page</option>
          </select>
        </div>
      </div>

      <!-- Row 3: Buttons -->
      <div class="action-buttons">
        <button @click="goToGenerate" class="next-button">Generate Slides</button>
        <button @click="goToPptToVideo" class="ppt-video-button">PPT2Presentation</button>
      </div>
    </div>
  </ElectricBorder>
</template>

<script>
import ElectricBorder from './ElectricBorder.vue'

export default {
  name: 'UploadComponent',
  components: {
    ElectricBorder
  },
  data() {
    return {
      pptxFile: null,
      pdfFile: null,
      url: '',
      topic: '',
      selectedPages: 6,
      pagesOptions: Array.from({ length: 12 }, (_, i) => i + 3),
      isPptxEnabled:true
    }
  },
  methods: {
    handleFileUpload(event, fileType) {
      console.log("file uploaded :", fileType)
      const file = event.target.files[0]
      if (fileType === 'pptx') {
        this.pptxFile = file
      } else if (fileType === 'pdf') {
        this.pdfFile = file
      }
    },
    async goToGenerate() {
      this.$axios.get('/')
        .then(response => {
          console.log("Backend is running", response.data);
        })
        .catch(error => {
          console.error(error);
          alert('Backend is not running or too busy, your task will not be processed');
          return;
        });

      if (!this.pdfFile && !this.url) {
        alert('Please upload a PDF file OR enter a URL.');
        return;
      }

      const formData = new FormData();
      if (this.pptxFile) {
        formData.append('pptxFile', this.pptxFile);
      }
      if (this.pdfFile) {
        formData.append('pdfFile', this.pdfFile);
      }
      if (this.url) {
        formData.append('url', this.url);
      }
      if (this.topic) {
        formData.append('topic', this.topic);
      }
      formData.append('numberOfPages', this.selectedPages);

      try {
        const uploadResponse = await this.$axios.post('/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        const taskId = uploadResponse.data.task_id
        console.log("Task ID:", taskId)
        // Navigate to Generate component with taskId
        this.$router.push({ name: 'Generate', state: { taskId: taskId } })
      } catch (error) {
        console.error("Upload error:", error)
        this.statusMessage = 'Failed to upload files.'
      }
    },
    goToPptToVideo() {
      this.$router.push({ name: 'PptToVideo' })
    }
  }
}
</script>

<style scoped>
.upload-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: auto; /* Fit content */
  min-height: 50vh;
  /* background-color: #f0f8ff; REMOVED to be transparent/glassy */
  background: rgba(255, 255, 255, 0.1); /* Glassmorphism */
  backdrop-filter: blur(10px);
  padding: 40px;
  box-sizing: border-box;
  border-radius: 16px; /* Match ElectricBorder */
  margin: 20px;
}

.upload-options {
  display: flex;
  flex-direction: column;
  gap: 30px;
  width: 100%;
  max-width: 800px;
}

.upload-buttons,
.selectors {
  display: flex;
  justify-content: center;
  gap: 20px;
  width: 100%;
}

.upload-section,
.pages-selection {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.upload-label {
  position: relative;
  background-color: rgba(66, 185, 131, 0.9); /* Slight opacity */
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  width: 100%;
  text-align: center;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
  font-size: 20px;
  box-sizing: border-box;
}

.upload-label:hover {
  background-color: #369870;
}

.upload-section input[type="file"] {
  display: none;
}

.pages-selection select {
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ccc;
  width: 100%;
  height: 40px;
  box-sizing: border-box;
  font-size: 16px;
  background: rgba(255, 255, 255, 0.9);
}

.next-button {
  background-color: #35495e;
  color: white;
  padding: 12px 0;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  width: 220px;
  font-size: 20px;
  font-weight: 700;
  transition: background-color 0.3s, transform 0.2s;
  box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

.next-button:hover {
  background-color: #2c3e50;
  transform: scale(1.05);
}

.ppt-video-button {
  background-color: #e74c3c;
  color: white;
  padding: 12px 0;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  width: 220px;
  font-size: 20px;
  font-weight: 700;
  transition: background-color 0.3s, transform 0.2s;
  box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

.ppt-video-button:hover {
  background-color: #c0392b;
  transform: scale(1.05);
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 30px;
}

.uploaded-symbol {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: #fff;
  font-size: 18px;
  text-shadow: 0 0 5px black;
}

@media (max-width: 600px) {

  .upload-buttons,
  .selectors {
    flex-direction: column;
    gap: 20px;
  }

  .action-buttons {
    flex-direction: column;
    gap: 15px;
  }

  .next-button,
  .ppt-video-button {
    width: 100%;
  }
}

.url-input, .topic-input {
  width: 100%;
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ccc;
  font-size: 16px;
  box-sizing: border-box;
  font-family: inherit;
  margin-top: 5px;
  background: rgba(255, 255, 255, 0.9);
}
</style>
