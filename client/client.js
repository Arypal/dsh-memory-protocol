// dsh-memory-protocol 客户端半边：记忆开关的 /dsh-memory/toggle 客户端。
// 2026-10-30 起：开关 UI 已迁到「记忆星图」面板（dsh-memory-graph），这里不再注册会话头部插槽。
//
// 设计：
//   - Apple iOS 风格开关：拨到右边 = 绿色 = 记忆开；拨到左边 = 红色 = 记忆关。
//   - 液态玻璃（liquid glass）：backdrop-filter 模糊 + 高光渐变 + 内阴影。
//   - 状态来自同源路由 GET /dsh-memory/toggle（服务端插件注册），
//     点击 POST 切换，写的是 memory/paused.json —— 与 pre-step / turn-stopping
//     读的同一个文件，所以点完 3 秒内生效。
//
// 加载方式：DSH 客户端通过 window.__ModuleLoader__ 加载本 bundle（package.json
// 的 dsh.client 字段），factory 返回 cordis 插件模块（name/inject/apply）。
window.__ModuleLoader__.load({
    // 注意：这里的 id 必须是 npm 包名（dsh-memory-protocol），宿主按包名
    // 生成 boot graph 行并用它 import；写成别的名字会让该行「未注册」，
    // 宿主回退到单资源 URL 再执行一次 bundle → duplicate factory registration。
    id: 'dsh-memory-protocol',
    factory: (require) => {
        const module = { exports: {} };
        const exports = module.exports;
        Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
        const react = require('react');
        const h = react.createElement;

        // ── 样式：Apple 开关 + 液态玻璃 ──────────────────────────────────
        const CSS = `
.dsh-mem-switch {
  position: relative;
  display: inline-flex;
  align-items: center;
  width: 46px;
  height: 26px;
  padding: 0;
  border: 0;
  border-radius: 13px;
  cursor: pointer;
  isolation: isolate;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(255,255,255,.30), rgba(255,255,255,.10));
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,.45),
    inset 0 1px 6px rgba(0,0,0,.28),
    0 2px 10px rgba(0,0,0,.22);
  backdrop-filter: blur(18px) saturate(180%);
  -webkit-backdrop-filter: blur(18px) saturate(180%);
  transition: background .38s cubic-bezier(.34,1.56,.64,1),
              box-shadow .38s ease, opacity .2s ease;
}
.dsh-mem-switch.is-on {
  background: linear-gradient(135deg, #4ade80, #10b981 70%);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,.55),
    inset 0 1px 4px rgba(255,255,255,.40),
    0 2px 12px rgba(16,185,129,.50);
}
.dsh-mem-switch.is-off {
  background: linear-gradient(135deg, #fb7185, #ef4444 70%);
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,.55),
    inset 0 1px 4px rgba(255,255,255,.40),
    0 2px 12px rgba(239,68,68,.50);
}
.dsh-mem-switch.is-busy { opacity: .55; cursor: progress; }
.dsh-mem-glare {
  position: absolute;
  inset: 0;
  border-radius: 13px;
  pointer-events: none;
  background: linear-gradient(118deg,
    rgba(255,255,255,.62) 0%,
    rgba(255,255,255,.12) 32%,
    rgba(255,255,255,0) 55%,
    rgba(255,255,255,.16) 78%,
    rgba(255,255,255,.05) 100%);
  mix-blend-mode: screen;
}
.dsh-mem-knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: radial-gradient(circle at 32% 24%, #ffffff 0%, #f2f5f9 55%, #d9e0e8 100%);
  box-shadow:
    0 1px 3px rgba(0,0,0,.38),
    0 0 0 .5px rgba(0,0,0,.10),
    inset 0 1px 1px rgba(255,255,255,.95),
    inset 0 -1px 2px rgba(0,0,0,.10);
  transition: transform .38s cubic-bezier(.34,1.56,.64,1);
}
.dsh-mem-switch.is-on  .dsh-mem-knob { transform: translateX(20px); }
.dsh-mem-switch.is-off .dsh-mem-knob { transform: translateX(0); }
`;
        const TAG_ID = 'dsh-memory-ui/switch.css';
        const injectCss = () => {
            try {
                if (typeof document === 'undefined')
                    return;
                const sel = 'style[data-plugin-css=' + JSON.stringify(TAG_ID) + ']';
                if (document.querySelector(sel))
                    return;
                const tag = document.createElement('style');
                tag.dataset.plugin = 'dsh-memory-ui';
                tag.dataset.pluginCss = TAG_ID;
                tag.textContent = CSS;
                document.head.appendChild(tag);
                const registry = (globalThis.__DSH_PLUGIN_CSS__ ??= []);
                if (!registry.some((e) => e && e.tagId === TAG_ID))
                    registry.push({ tagId: TAG_ID, css: CSS });
            }
            catch {
                // 注入样式失败不影响功能（只是没有动画）
            }
        };

        // ── 组件：Apple 胶囊开关 ─────────────────────────────────────────
        function MemorySwitch() {
            // paused=null 表示还没拿到状态（首次渲染）；false=记忆开；true=记忆关
            const [paused, setPaused] = react.useState(null);
            const [busy, setBusy] = react.useState(false);
            react.useEffect(() => {
                let alive = true;
                const load = () => {
                    fetch('/dsh-memory/toggle', { method: 'GET' })
                        .then((r) => r.json())
                        .then((s) => {
                        if (alive && s && typeof s.paused === 'boolean')
                            setPaused(s.paused);
                    })
                        .catch(() => { });
                };
                load();
                const timer = setInterval(load, 5000);
                return () => {
                    alive = false;
                    clearInterval(timer);
                };
            }, []);
            const toggle = () => {
                if (busy || paused === null)
                    return;
                const nextPaused = !paused;
                setBusy(true);
                setPaused(nextPaused); // 乐观更新：拨动立即反馈
                fetch('/dsh-memory/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: nextPaused ? 'pause' : 'resume', scope: 'all' }),
                })
                    .then((r) => r.json())
                    .then((s) => {
                    if (s && typeof s.paused === 'boolean')
                        setPaused(s.paused);
                })
                    .catch(() => { })
                    .finally(() => setBusy(false));
            };
            const stateClass = paused === null ? '' : (paused ? 'is-off' : 'is-on');
            const title = paused === null
                ? '记忆状态加载中…'
                : (paused ? '记忆已关闭（点击开启）' : '记忆已开启（点击关闭）');
            return h('button', {
                type: 'button',
                className: 'dsh-mem-switch ' + stateClass + (busy ? ' is-busy' : ''),
                title,
                'aria-label': title,
                onClick: toggle,
            }, h('span', { className: 'dsh-mem-glare' }), h('span', { className: 'dsh-mem-knob' }));
        }

        // ── cordis 插件接线 ──────────────────────────────────────────────
        const NAME = 'dsh-memory-ui';
        const inject = ['slots'];
        function apply(ctx) {
            try {
                injectCss();
            }
            catch (err) {
                // 样式失败不影响开关功能
                console.warn('dsh-memory-ui: css inject failed', err);
            }
            // ── 2026-10-30 迁移：记忆开关已移到「记忆星图」面板（dsh-memory-graph）──
            // 不再注册会话头部插槽。MemorySwitch 组件与 /dsh-memory/toggle 逻辑保留，
            // 需要恢复时取消下面整段注释即可。
            // try {
            //     ctx.slots.inject('conversation.session.header.actions', () => ctx.slots.register({
            //         name: 'conversation.session.header.actions',
            //         id: NAME,
            //         order: 100,
            //     }, () => h(MemorySwitch)));
            // }
            // catch (err) {
            //     console.warn('dsh-memory-ui: slot register failed', err);
            // }
        }
        exports.apply = apply;
        exports.inject = inject;
        exports.name = NAME;
        return module.exports;
    },
});
