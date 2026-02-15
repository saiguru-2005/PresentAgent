<template>
  <div 
    ref="rootRef"
    :class="['electric-border', className]" 
    :style="{ ...styleVars, ...style }"
  >
    <svg ref="svgRef" class="eb-svg" aria-hidden="true" focusable="false">
      <defs>
        <filter :id="filterId" colorInterpolationFilters="sRGB" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="turbulence" baseFrequency="0.02" numOctaves="10" result="noise1" seed="1" />
          <feOffset in="noise1" dx="0" dy="0" result="offsetNoise1">
            <animate attributeName="dy" values="700; 0" dur="6s" repeatCount="indefinite" calcMode="linear" />
          </feOffset>

          <feTurbulence type="turbulence" baseFrequency="0.02" numOctaves="10" result="noise2" seed="1" />
          <feOffset in="noise2" dx="0" dy="0" result="offsetNoise2">
            <animate attributeName="dy" values="0; -700" dur="6s" repeatCount="indefinite" calcMode="linear" />
          </feOffset>

          <feTurbulence type="turbulence" baseFrequency="0.02" numOctaves="10" result="noise1" seed="2" />
          <feOffset in="noise1" dx="0" dy="0" result="offsetNoise3">
            <animate attributeName="dx" values="490; 0" dur="6s" repeatCount="indefinite" calcMode="linear" />
          </feOffset>

          <feTurbulence type="turbulence" baseFrequency="0.02" numOctaves="10" result="noise2" seed="2" />
          <feOffset in="noise2" dx="0" dy="0" result="offsetNoise4">
            <animate attributeName="dx" values="0; -490" dur="6s" repeatCount="indefinite" calcMode="linear" />
          </feOffset>

          <feComposite in="offsetNoise1" in2="offsetNoise2" result="part1" />
          <feComposite in="offsetNoise3" in2="offsetNoise4" result="part2" />
          <feBlend in="part1" in2="part2" mode="color-dodge" result="combinedNoise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="combinedNoise"
            scale="30"
            xChannelSelector="R"
            yChannelSelector="B"
          />
        </filter>
      </defs>
    </svg>

    <div class="eb-layers">
      <div ref="strokeRef" class="eb-stroke" />
      <div class="eb-glow-1" />
      <div class="eb-glow-2" />
      <div class="eb-background-glow" />
    </div>

    <div class="eb-content">
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
/* eslint-disable no-undef */
import { ref, onMounted, onUnmounted, computed, watch } from 'vue';

const props = defineProps({
  color: {
    type: String,
    default: '#5227FF'
  },
  speed: {
    type: Number,
    default: 1
  },
  chaos: {
    type: Number,
    default: 1
  },
  thickness: {
    type: Number,
    default: 2
  },
  className: {
    type: String,
    default: ''
  },
  style: {
    type: Object,
    default: () => ({})
  }
});

const generateId = () => Math.random().toString(36).substr(2, 9);
const rawId = generateId();
const filterId = `turbulent-displace-${rawId}`;

const rootRef = ref(null);
const svgRef = ref(null);
const strokeRef = ref(null);
let resizeObserver = null;

const styleVars = computed(() => ({
  '--electric-border-color': props.color,
  '--eb-border-width': `${props.thickness}px`
}));

const updateAnim = () => {
  const svg = svgRef.value;
  const host = rootRef.value;
  if (!svg || !host) return;

  if (strokeRef.value) {
    strokeRef.value.style.filter = `url(#${filterId})`;
  }

  const width = Math.max(1, Math.round(host.clientWidth || host.getBoundingClientRect().width || 0));
  const height = Math.max(1, Math.round(host.clientHeight || host.getBoundingClientRect().height || 0));

  const dyAnims = Array.from(svg.querySelectorAll('feOffset > animate[attributeName="dy"]'));
  if (dyAnims.length >= 2) {
    dyAnims[0].setAttribute('values', `${height}; 0`);
    dyAnims[1].setAttribute('values', `0; -${height}`);
  }

  const dxAnims = Array.from(svg.querySelectorAll('feOffset > animate[attributeName="dx"]'));
  if (dxAnims.length >= 2) {
    dxAnims[0].setAttribute('values', `${width}; 0`);
    dxAnims[1].setAttribute('values', `0; -${width}`);
  }

  const baseDur = 6;
  const dur = Math.max(0.001, baseDur / (props.speed || 1));
  [...dyAnims, ...dxAnims].forEach(a => a.setAttribute('dur', `${dur}s`));

  const disp = svg.querySelector('feDisplacementMap');
  if (disp) disp.setAttribute('scale', String(30 * (props.chaos || 1)));

  const filterEl = svg.querySelector(`#${CSS.escape(filterId)}`);
  if (filterEl) {
    filterEl.setAttribute('x', '-200%');
    filterEl.setAttribute('y', '-200%');
    filterEl.setAttribute('width', '500%');
    filterEl.setAttribute('height', '500%');
  }

  requestAnimationFrame(() => {
    [...dyAnims, ...dxAnims].forEach(a => {
      // Logic for beginElement might be less reliable in Vue depending on DOM state, 
      // but standard SVG API should work.
      if (typeof a.beginElement === 'function') {
        try {
          a.beginElement();
        } catch (e) {
          console.warn('ElectricBorder: beginElement failed', e);
        }
      }
    });
  });
};

watch(() => [props.speed, props.chaos], () => {
  updateAnim();
});

onMounted(() => {
  if (rootRef.value) {
     resizeObserver = new ResizeObserver(() => updateAnim());
     resizeObserver.observe(rootRef.value);
     updateAnim();
  }
});

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect();
});
</script>

<style scoped>
.electric-border {
  --electric-light-color: oklch(from var(--electric-border-color) l c h);
  /* --eb-border-width is set via inline style */
  position: relative;
  border-radius: inherit;
  overflow: visible;
  isolation: isolate;
}

.eb-svg {
  position: fixed;
  left: -10000px;
  top: -10000px;
  width: 10px;
  height: 10px;
  opacity: 0.001;
  pointer-events: none;
}

.eb-content {
  position: relative;
  border-radius: inherit;
  z-index: 1;
}

.eb-layers {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  z-index: 2;
}

.eb-stroke,
.eb-glow-1,
.eb-glow-2,
.eb-background-glow {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  box-sizing: border-box;
}

.eb-stroke {
  border: var(--eb-border-width) solid var(--electric-border-color);
}

.eb-glow-1 {
  border: var(--eb-border-width) solid oklch(from var(--electric-border-color) l c h / 0.6);
  opacity: 0.5;
  filter: blur(calc(0.5px + (var(--eb-border-width) * 0.25)));
}

.eb-glow-2 {
  border: var(--eb-border-width) solid var(--electric-light-color);
  opacity: 0.5;
  filter: blur(calc(2px + (var(--eb-border-width) * 0.5)));
}

.eb-background-glow {
  z-index: -1;
  transform: scale(1.08);
  filter: blur(32px);
  opacity: 0.3;
  background: linear-gradient(-30deg, var(--electric-light-color), transparent, var(--electric-border-color));
}
</style>
