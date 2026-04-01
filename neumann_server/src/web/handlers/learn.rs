// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grokking experiment dashboard.
//!
//! Renders a Poincare disk showing learned embeddings for modular
//! arithmetic, with train/test accuracy curves that reveal the
//! grokking phase transition.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use maud::{html, Markup, PreEscaped};
use serde::Deserialize;

use crate::web::templates::{layout, m_empty, m_header};
use crate::web::{AdminContext, NavItem};

/// Main dashboard page.
pub async fn dashboard(State(ctx): State<Arc<AdminContext>>) -> Markup {
    if ctx.grok.is_none() && ctx.training.is_none() {
        return layout(
            "Learn",
            NavItem::Learn,
            m_empty(
                "No Experiment",
                "Start with NEUMANN_LEARN=1 to enable the grokking dashboard.",
            ),
        );
    }

    let content = html! {
        (m_header("Hyperbolic Grokking", Some("Modular arithmetic a + b mod p on the Poincare disk")))

        // Stats
        div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4" {
            @for (id, label) in [
                ("stat-status", "Status"), ("stat-step", "Step"),
                ("stat-train-loss", "Train Loss"), ("stat-test-loss", "Test Loss"),
                ("stat-train-acc", "Train Acc"), ("stat-test-acc", "Test Acc"),
            ] {
                div class="m-card" {
                    div class="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider mb-1" { (label) }
                    div id=(id) class="text-base font-mono text-[var(--text-primary)]" { "-" }
                }
            }
        }

        // Controls
        div class="flex flex-wrap gap-2 mb-4 items-end" {
            button class="m-btn" onclick="trainAction('start')" { "Start" }
            button class="m-btn" onclick="trainAction('pause')" { "Pause" }
            button class="m-btn" onclick="doSingleStep()" { "Step" }
            div class="w-px h-6 bg-[var(--border)] mx-2" {}
            @for (id, label, val, width) in [
                ("in-prime", "p", "31", "w-14"),
                ("in-hidden", "Hidden", "128", "w-16"),
                ("in-lr", "LR", "0.001", "w-20"),
                ("in-wd", "WD", "1.0", "w-14"),
                ("in-steps", "Steps", "50000", "w-20"),
                ("in-seed", "Seed", "42", "w-14"),
            ] {
                div {
                    label class="text-[10px] text-[var(--text-tertiary)] uppercase block mb-0.5" { (label) }
                    input id=(id) type="number" value=(val)
                        class={(width) " px-1.5 py-1 rounded bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border)] font-mono text-xs"} {}
                }
            }
            button class="m-btn" onclick="resetSession()" { "Reset" }
        }

        // Main grid: disk + charts
        div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4" {
            div class="m-card p-2" {
                div class="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider mb-1 px-1" { "Embeddings (Poincare Disk)" }
                canvas id="poincare-canvas" width="520" height="520"
                    style="display: block; margin: 0 auto; background: #111; border-radius: 6px;" {}
            }
            div class="flex flex-col gap-4" {
                div class="m-card p-2" {
                    div class="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider mb-1 px-1" { "Accuracy (train vs test)" }
                    canvas id="acc-chart" width="520" height="220"
                        style="display: block; margin: 0 auto; background: #111; border-radius: 6px;" {}
                }
                div class="m-card p-2" {
                    div class="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider mb-1 px-1" { "Loss (train vs test)" }
                    canvas id="loss-chart" width="520" height="220"
                        style="display: block; margin: 0 auto; background: #111; border-radius: 6px;" {}
                }
            }
        }

        // Log
        div class="m-card p-2 mb-4" {
            div class="text-[10px] text-[var(--text-tertiary)] uppercase tracking-wider mb-1 px-1" { "Training Log" }
            pre id="log-output"
                class="overflow-y-auto font-mono text-[11px] leading-relaxed text-[var(--text-secondary)] bg-[#111] rounded p-2"
                style="max-height: 200px; white-space: pre;" {
                "Ready. Click Start to begin grokking experiment.\n"
            }
        }

        (PreEscaped(LEARN_SCRIPT))
    };

    layout("Learn", NavItem::Learn, content)
}

/// Viz data endpoint.
pub async fn viz_data(State(ctx): State<Arc<AdminContext>>) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        return Json(grok.read().to_viz_data()).into_response();
    }
    if let Some(ref training) = ctx.training {
        let result = training.read().to_viz_data();
        return match result {
            Ok(data) => Json(data).into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response(),
        };
    }
    StatusCode::NOT_FOUND.into_response()
}

/// Status endpoint.
pub async fn status(State(ctx): State<Arc<AdminContext>>) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        return Json(grok.read().stats()).into_response();
    }
    if let Some(ref training) = ctx.training {
        return Json(training.read().stats()).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

/// Start training.
pub async fn start(State(ctx): State<Arc<AdminContext>>) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        grok.write().start();
        return StatusCode::OK;
    }
    if let Some(ref training) = ctx.training {
        training.write().start();
        return StatusCode::OK;
    }
    StatusCode::NOT_FOUND
}

/// Pause training.
pub async fn pause(State(ctx): State<Arc<AdminContext>>) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        grok.write().pause();
        return StatusCode::OK;
    }
    if let Some(ref training) = ctx.training {
        let mut session = training.write();
        if session.status() == tensor_learn::TrainingStatus::Paused {
            session.resume();
        } else {
            session.pause();
        }
        drop(session);
        return StatusCode::OK;
    }
    StatusCode::NOT_FOUND
}

/// Step endpoint.
pub async fn step(State(ctx): State<Arc<AdminContext>>) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        grok.write().step();
        return Json(grok.read().stats()).into_response();
    }
    ctx.training.as_ref().map_or_else(
        || StatusCode::NOT_FOUND.into_response(),
        |training| {
            let mut session = training.write();
            match session.step() {
                Ok(()) => Json(session.stats()).into_response(),
                Err(err) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": err.to_string()})),
                )
                    .into_response(),
            }
        },
    )
}

/// Reset parameters.
#[derive(Deserialize)]
pub struct ResetParams {
    /// Prime modulus.
    prime: Option<usize>,
    /// Embedding dimension.
    embed_dim: Option<usize>,
    /// Hidden dim.
    hidden_dim: Option<usize>,
    /// Learning rate.
    learning_rate: Option<f64>,
    /// Weight decay.
    weight_decay: Option<f64>,
    /// Total steps.
    total_steps: Option<usize>,
    /// Seed.
    seed: Option<u64>,
}

/// Reset with new grokking session.
pub async fn reset(
    State(ctx): State<Arc<AdminContext>>,
    Json(params): Json<ResetParams>,
) -> impl IntoResponse {
    if let Some(ref grok) = ctx.grok {
        let config = tensor_learn::GrokConfig {
            prime: params.prime.unwrap_or(31),
            embed_dim: params.embed_dim.unwrap_or(32),
            hidden_dim: params.hidden_dim.unwrap_or(128),
            learning_rate: params.learning_rate.unwrap_or(1e-3),
            weight_decay: params.weight_decay.unwrap_or(1.0),
            total_steps: params.total_steps.unwrap_or(50_000),
            seed: params.seed.unwrap_or(42),
            ..tensor_learn::GrokConfig::default()
        };
        *grok.write() = tensor_learn::GrokSession::new(config);
        return StatusCode::OK.into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}

/// Dashboard JavaScript.
const LEARN_SCRIPT: &str = r"<script>
(function() {
    const diskC = document.getElementById('poincare-canvas');
    const diskX = diskC.getContext('2d');
    const accC  = document.getElementById('acc-chart');
    const accX  = accC.getContext('2d');
    const lossC = document.getElementById('loss-chart');
    const lossX = lossC.getContext('2d');
    const logEl = document.getElementById('log-output');
    let vizData = null, pollTimer = null, lastStep = -1;

    const el = id => document.getElementById(id);
    const pad = (n,w) => String(n).padStart(w);
    const f = (v,d) => v.toFixed(d);
    function log(m) { logEl.textContent += m + '\n'; logEl.scrollTop = logEl.scrollHeight; }

    // -- Poincare disk --
    function drawDisk() {
        const w = diskC.width, h = diskC.height, cx = w/2, cy = h/2, r = Math.min(cx,cy)-16;
        diskX.clearRect(0,0,w,h);
        diskX.beginPath(); diskX.arc(cx,cy,r,0,2*Math.PI);
        diskX.strokeStyle='#333'; diskX.lineWidth=1.5; diskX.stroke();
        for(let i=1;i<=4;i++){diskX.beginPath();diskX.arc(cx,cy,r*i/5,0,2*Math.PI);diskX.strokeStyle='#1a1a1a';diskX.lineWidth=0.5;diskX.stroke();}
        if(!vizData||!vizData.nodes.length)return;
        const p = vizData.nodes.length;
        for(const n of vizData.nodes){
            const x=cx+n.x*r, y=cy-n.y*r;
            const hue = (parseInt(n.id)/p)*360;
            diskX.beginPath(); diskX.arc(x,y,4,0,2*Math.PI);
            diskX.fillStyle='hsl('+hue+',80%,60%)'; diskX.fill();
            diskX.font='9px monospace'; diskX.fillStyle='#666'; diskX.textAlign='center';
            diskX.fillText(n.id, x, y-7);
        }
    }

    // -- Chart renderer --
    function drawChart(ctx, canvas, series, yLabel, isAcc) {
        const w=canvas.width, h=canvas.height;
        const P={l:48,r:8,t:14,b:22}, pw=w-P.l-P.r, ph=h-P.t-P.b;
        ctx.clearRect(0,0,w,h);
        if(!series.length||!series[0].data.length){
            ctx.fillStyle='#444';ctx.font='12px monospace';ctx.textAlign='center';
            ctx.fillText('Awaiting data...',w/2,h/2); return;
        }
        const allVals=[]; series.forEach(s=>s.data.forEach(v=>allVals.push(v)));
        let mn=Math.min(...allVals), mx=Math.max(...allVals);
        if(isAcc){mn=0;mx=1;}
        const rng=mx-mn||1;
        const maxLen=Math.max(...series.map(s=>s.data.length));
        const toX=i=>P.l+(i/(maxLen-1||1))*pw;
        const toY=v=>P.t+((mx-v)/rng)*ph;

        // Grid
        ctx.strokeStyle='#222';ctx.lineWidth=0.5;
        for(let i=0;i<=4;i++){const y=P.t+ph*i/4;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(w-P.r,y);ctx.stroke();}

        // Series
        series.forEach(s=>{
            ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=1.5;
            for(let i=0;i<s.data.length;i++){
                const x=toX(i),y=toY(s.data[i]);
                i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
            }
            ctx.stroke();
        });

        // Labels
        ctx.fillStyle='#555';ctx.font='9px monospace';
        ctx.textAlign='right'; ctx.fillText(isAcc?(mx*100).toFixed(0)+'%':f(mx,2),P.l-4,P.t+8);
        ctx.fillText(isAcc?(mn*100).toFixed(0)+'%':f(mn,2),P.l-4,P.t+ph+3);
        ctx.textAlign='left';
        // Legend
        let lx=P.l+6;
        series.forEach(s=>{
            ctx.fillStyle=s.color;ctx.fillRect(lx,P.t+2,10,3);
            ctx.fillStyle='#666';ctx.fillText(s.label,lx+14,P.t+8);
            lx+=60;
        });
    }

    async function fetchViz(){try{const r=await fetch('/learn/viz');if(r.ok){vizData=await r.json();drawDisk();}}catch(_){}}

    function updateUI(st) {
        const sn={Idle:'IDLE',Running:'RUN',Paused:'PAUSE',Completed:'DONE'};
        el('stat-status').textContent=sn[st.status]||st.status;
        el('stat-step').textContent=st.step+'/'+st.total_steps;
        el('stat-train-loss').textContent=st.train_loss<1e300?f(st.train_loss,4):'-';
        el('stat-test-loss').textContent=st.test_loss<1e300?f(st.test_loss,4):'-';
        el('stat-train-acc').textContent=st.train_acc!==undefined?(st.train_acc*100).toFixed(1)+'%':'-';
        el('stat-test-acc').textContent=st.test_acc!==undefined?(st.test_acc*100).toFixed(1)+'%':'-';

        if(st.train_acc_history){
            drawChart(accX,accC,[
                {data:st.train_acc_history,color:'#3b82f6',label:'train'},
                {data:st.test_acc_history,color:'#ef4444',label:'test'},
            ],'Accuracy',true);
        }
        if(st.train_loss_history){
            drawChart(lossX,lossC,[
                {data:st.train_loss_history,color:'#3b82f6',label:'train'},
                {data:st.test_loss_history,color:'#ef4444',label:'test'},
            ],'Loss',false);
        }
    }

    function logStep(st){
        if(st.step<=lastStep)return;
        lastStep=st.step;
        let s=pad(st.step,6)+'/'+st.total_steps;
        s+='  trL='+f(st.train_loss,3)+' teL='+f(st.test_loss,3);
        s+='  trA='+(st.train_acc*100).toFixed(1)+'%';
        s+='  teA='+(st.test_acc*100).toFixed(1)+'%';
        log(s);
    }

    async function trainAction(action){
        try{
            if(action==='start'){
                const sr=await fetch('/learn/status');
                if(sr.ok){const st=await sr.json();if(st.status==='Completed'){await doReset();}}
            }
            await fetch('/learn/'+action,{method:'POST'});
            await fetchViz();
            const sr=await fetch('/learn/status');
            if(sr.ok){const st=await sr.json();updateUI(st);managePoll(st.status);}
        }catch(_){}
    }

    async function doSingleStep(){
        try{
            const r=await fetch('/learn/step',{method:'POST'});
            await fetchViz();
            if(r.ok){const st=await r.json();updateUI(st);logStep(st);}
        }catch(_){}
    }

    function getParams(){return{
        prime:parseInt(el('in-prime').value)||31,
        hidden_dim:parseInt(el('in-hidden').value)||128,
        learning_rate:parseFloat(el('in-lr').value)||1e-3,
        weight_decay:parseFloat(el('in-wd').value)||1.0,
        total_steps:parseInt(el('in-steps').value)||50000,
        seed:parseInt(el('in-seed').value)||42,
    };}

    async function doReset(){
        await fetch('/learn/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getParams())});
        lastStep=-1;
    }

    async function resetSession(){
        const p=getParams();
        logEl.textContent='';
        log('--- Reset: p='+p.prime+' h='+p.hidden_dim+' lr='+p.learning_rate+' wd='+p.weight_decay+' steps='+p.total_steps+' seed='+p.seed+' ---');
        await doReset(); await fetchViz();
        const sr=await fetch('/learn/status');if(sr.ok)updateUI(await sr.json());
    }

    function managePoll(status){
        if(status==='Running'){
            if(!pollTimer){pollTimer=setInterval(async()=>{
                await fetch('/learn/step',{method:'POST'});
                await fetchViz();
                const sr=await fetch('/learn/status');
                if(sr.ok){const st=await sr.json();updateUI(st);logStep(st);
                    if(st.status!=='Running'){clearInterval(pollTimer);pollTimer=null;
                        log('--- Done. train_acc='+(st.train_acc*100).toFixed(1)+'% test_acc='+(st.test_acc*100).toFixed(1)+'% ---');}}
            },300);}
        }else{if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}
    }

    window.trainAction=trainAction;
    window.doSingleStep=doSingleStep;
    window.resetSession=resetSession;
    fetchViz();
    drawChart(accX,accC,[],'',true);
    drawChart(lossX,lossC,[],'',false);
})();
</script>";
