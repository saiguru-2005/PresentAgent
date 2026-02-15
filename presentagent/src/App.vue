<template>
  <div id="app">
    <Hyperspeed class="bg-animation" />
    <ClickSpark class="click-wrapper">
      <div class="content-overlay">
        <header class="app-header">
          <h1>PresentAgent</h1>
        </header>
        <main class="app-main">
          <router-view />
        </main>
      </div>
    </ClickSpark>
  </div>
</template>

<script>
import Hyperspeed from './components/HyperspeedBg.vue'
import ClickSpark from './components/ClickSpark.vue'

export default {
  name: 'App',
  components: {
    Hyperspeed,
    ClickSpark
  }
}
</script>

<style>
/* Global reset for full screen */
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden; /* Hyperspeed handles overflow */
}
</style>

<style scoped>
#app {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background-color: #000; /* Fallback for 3D */
}

/* Background Animation Layer */
.bg-animation {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

/* Interaction Layer */
.click-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  z-index: 1;
}

/* Content Container to center everything */
.content-overlay {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  pointer-events: none; /* Let clicks pass to ClickSpark mostly, re-enable for buttons */
}

.app-header {
  font-family: 'American Typewriter', serif;
  text-align: center;
  font-weight: bold;
  font-size: 24px;
  height: 10vh;
  color: #fff; /* White text for dark background */
  text-shadow: 0 0 10px rgba(255,255,255,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto; /* Enable header interactions */
  background: rgba(0,0,0,0.2); /* Slight dim */
}

.app-main {
  flex: 1;
  overflow: auto; /* Allow scrolling content */
  font-family: "Arial Rounded MT Bold";
  pointer-events: auto; /* Enable form interactions */
  display: flex; /* Center router view */
  align-items: center;
  justify-content: center;
}
</style>
