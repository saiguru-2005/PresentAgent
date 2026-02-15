<template>
  <div 
    @click="handleClick" 
    style="position: relative; width: 100%; height: 100%;"
  >
    <canvas
      ref="canvasRef"
      style="
        width: 100%;
        height: 100%;
        display: block;
        user-select: none;
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
      "
    />
    <slot></slot>
  </div>
</template>

<script setup>
/* eslint-disable no-undef */
import { ref, onMounted, onUnmounted } from 'vue';

const props = defineProps({
  sparkColor: {
    type: String,
    default: '#fff'
  },
  sparkSize: {
    type: Number,
    default: 10
  },
  sparkRadius: {
    type: Number,
    default: 15
  },
  sparkCount: {
    type: Number,
    default: 8
  },
  duration: {
    type: Number,
    default: 400
  },
  easing: {
    type: String,
    default: 'ease-out'
  },
  extraScale: {
    type: Number,
    default: 1.0
  }
});

const canvasRef = ref(null);
const sparks = ref([]); // Using simplified array, not ref of ref
const startTime = ref(null);
let animationId = null;
let resizeTimeout = null;
let resizeObserver = null;

const easeFunc = (t) => {
  switch (props.easing) {
    case 'linear':
      return t;
    case 'ease-in':
      return t * t;
    case 'ease-in-out':
      return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    default:
      return t * (2 - t);
  }
};

const draw = (timestamp) => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (!startTime.value) {
    startTime.value = timestamp;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  sparks.value = sparks.value.filter(spark => {
    const elapsed = timestamp - spark.startTime;
    if (elapsed >= props.duration) {
      return false;
    }

    const progress = elapsed / props.duration;
    const eased = easeFunc(progress);

    const distance = eased * props.sparkRadius * props.extraScale;
    const lineLength = props.sparkSize * (1 - eased);

    const x1 = spark.x + distance * Math.cos(spark.angle);
    const y1 = spark.y + distance * Math.sin(spark.angle);
    const x2 = spark.x + (distance + lineLength) * Math.cos(spark.angle);
    const y2 = spark.y + (distance + lineLength) * Math.sin(spark.angle);

    ctx.strokeStyle = props.sparkColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    return true;
  });

  animationId = requestAnimationFrame(draw);
};

const resizeCanvas = () => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const parent = canvas.parentElement;
  if (!parent) return;

  const { width, height } = parent.getBoundingClientRect();
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
};

const handleResize = () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(resizeCanvas, 100);
};

const handleClick = (e) => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const now = performance.now();
  const newSparks = Array.from({ length: props.sparkCount }, (_, i) => ({
    x,
    y,
    angle: (2 * Math.PI * i) / props.sparkCount,
    startTime: now
  }));

  sparks.value.push(...newSparks);
};

onMounted(() => {
  const canvas = canvasRef.value;
  if (canvas) {
    const parent = canvas.parentElement;
    if (parent) {
       resizeObserver = new ResizeObserver(handleResize);
       resizeObserver.observe(parent);
       resizeCanvas();
    }
  }

  animationId = requestAnimationFrame(draw);
});

onUnmounted(() => {
  if (animationId) cancelAnimationFrame(animationId);
  if (resizeObserver) resizeObserver.disconnect();
  clearTimeout(resizeTimeout);
});
</script>
