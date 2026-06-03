// pi-kuhul-engine.js — π-KUHUL Field RAG Engine (ECMAScript port)
//
// THIS IS A RAG ENGINE where the retrieval space is curved:
//   Corpus    = Token states on 100D manifold (the "documents")
//   Retrieval = Geodesic phase interference (arccos, not cosine)
//   Augment   = Shard transitions by phase coherence (not doc stuffing)
//   Generate  = Field evolution step (wave propagation = generation)
//
// THE HAAB MONTHS = K'UHUL PHASES (not coincidence):
//   Haab months: Pop Wo Sip Sotz Sek Xul Yaxk'in Mol Ch'en Yax Sak Keh
//                Mak K'ank'in Muwan Pax K'ayab Kumk'u Wayeb
//   K'UHUL exec: Pop →      Wo →                Sek →     Ch'en → Xul
//   The calendar IS the execution schedule.
//
// 52 cards = SMGM-16 CardSlots = Calendar Round (52 Haab years = 18,980 days)
// Token.phase = OpticalNode.sh[] (SH wave state per band)
// shard_id = optical cluster membership (which icosphere region)
// interference() = our arccos(q·k^T) geodesic attention
// coherence_history = EntropyField values (high incoherence = high entropy)
//
// Connection to existing stack:
//   to_optical_nodes() → feed into OpticalProcessor / SVG3DComputeGraph
//   Token.long_count   → ARC replay tick counter
//   CardField.pi_mod   → pressure per fold (fold pressure mapper)
//   shard_transition   → ARC recorded between shards = ReplayableArc

const TAU = 2 * Math.PI;

// ─── Mayan digit system ───────────────────────────────────────────────────────

export const MAYAN_DIGITS = {
    '•':1,'••':2,'•••':3,'••••':4,
    '⎯':5,'⎯⎯':10,'⎯⎯⎯':15,
    '•⎯':6,'••⎯':7,'•••⎯':8,'••••⎯':9,
    '•⎯⎯':11,'••⎯⎯':12,'•••⎯⎯':13,'••••⎯⎯':14,
    '•⎯⎯⎯':16,'••⎯⎯⎯':17,'•••⎯⎯⎯':18,'••••⎯⎯⎯':19,
    '𑂐':0,'🐚':0,'○':0
};

const REVERSE_MAYAN = ['𑂐','•','••','•••','••••','⎯',
    '•⎯','••⎯','•••⎯','••••⎯','⎯⎯','•⎯⎯','••⎯⎯','•••⎯⎯','••••⎯⎯','⎯⎯⎯',
    '•⎯⎯⎯','••⎯⎯⎯','•••⎯⎯⎯','••••⎯⎯⎯'];

export function mayanToNumber(g) {
    if (MAYAN_DIGITS[g] !== undefined) return MAYAN_DIGITS[g];
    let v=0; for (const c of g) v=v*20+(MAYAN_DIGITS[c]??0); return v;
}
export function numberToMayan(n) {
    if (!Number.isSafeInteger(n)||n<0) return '𑂐';
    if (n===0) return '𑂐';
    const d=[]; let r=n;
    while(r>0){d.unshift(r%20);r=Math.floor(r/20);}
    return d.map(x=>REVERSE_MAYAN[x]).join('');
}

// ─── Maya calendar ────────────────────────────────────────────────────────────

// Haab months = K'UHUL phases. The calendar IS the execution schedule.
export const HAAB_MONTHS = [
    'Pop','Wo','Sip','Sotz','Sek','Xul',"Yaxk'in",'Mol',
    "Ch'en",'Yax','Sak','Keh','Mak',"K'ank'in",'Muwan',
    'Pax',"K'ayab","Kumk'u",'Wayeb'
];

export const MayanCalendar = {
    longCountToDays(b,k,t,u,kin){ return b*144000+k*7200+t*360+u*20+kin; },
    daysToLongCount(days){
        const b=Math.floor(days/144000); days%=144000;
        const k=Math.floor(days/7200);   days%=7200;
        const t=Math.floor(days/360);    days%=360;
        const u=Math.floor(days/20);     const kin=days%20;
        return {baktun:b,katun:k,tun:t,uinal:u,kin};
    },
    tzolkin(days){ return {number:(days%260%13)+1, dayIdx:days%260%20}; },
    haab(days){ return {day:days%365%20, month:HAAB_MONTHS[Math.floor(days%365/20)]}; },
    advance(lc, n=1){ return MayanCalendar.daysToLongCount(
        MayanCalendar.longCountToDays(lc.baktun,lc.katun,lc.tun,lc.uinal,lc.kin)+n); },
};

// ─── π-Geometry ───────────────────────────────────────────────────────────────

export const PiGeometry = {
    normalize(p){ const n=p%TAU; return n<0?n+TAU:n; },
    interference(p1,p2,a1=1,a2=1){
        const d=Math.abs(p1-p2)%TAU;
        return Math.cos(Math.min(d,TAU-d))*a1*a2;
    },
    rotate(phase,angle){ return PiGeometry.normalize(phase+angle); },
};

// ─── MayanMath ────────────────────────────────────────────────────────────────

export const MayanMath = {
    VIGESIMAL: 20,
    VENUS_SYNODIC: 584,
    CALENDAR_ROUND: 18980,
    ...MayanCalendar,
    normalizePhase: PiGeometry.normalize,
    normalizeTensor(t,scale=20){
        const n=Math.hypot(...t)||Number.EPSILON;
        return t.map(v=>v/n*scale);
    },
    coherence(phases,amps=null){
        let total=0; const n=phases.length;
        for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){
            const d=Math.abs(phases[i]-phases[j])%TAU;
            total+=2*(amps?amps[i]*amps[j]:1)*Math.cos(Math.min(d,TAU-d));
        }
        return total/(n*(n-1)||1);
    },
};

// ─── PiKuhulFieldEngine ───────────────────────────────────────────────────────

export class PiKuhulFieldEngine {
    constructor({numTokens=256,numCards=52,dimensions=16,dt=0.01,sigma=0.15,temperature=0.2}={}) {
        this.numTokens=numTokens; this.numCards=numCards; this.dimensions=dimensions;
        this.dt=dt; this.sigma=sigma; this.temperature=temperature;
        this.tokens=[]; this.cards=[];
        this.piTime=0; this.tick=0;
        this.globalLC={baktun:13,katun:0,tun:0,uinal:0,kin:0};
        this.coherenceHistory=[]; this.shardTransitions=0;
        this._init();
    }

    _init(){
        for(let i=0;i<this.numTokens;i++){
            const pos=Array.from({length:this.dimensions},()=>(Math.random()-0.5)*4);
            const vel=Array.from({length:this.dimensions},()=>(Math.random()-0.5)*0.2);
            this.tokens.push({pos,vel,phase:Math.random()*TAU,coherence:0,
                shardId:Math.floor(Math.random()*this.numCards),
                lc:{baktun:Math.floor(Math.random()*20),katun:Math.floor(Math.random()*20),
                    tun:Math.floor(Math.random()*20),uinal:Math.floor(Math.random()*18),
                    kin:Math.floor(Math.random()*20)}});
        }
        for(let i=0;i<this.numCards;i++){
            const adj=Array.from({length:8},(_,j)=>(i+j+1)%this.numCards);
            this.cards.push({amplitude:Array.from({length:4},()=>Math.random()*2-1),
                gradient:Array.from({length:4},()=>Math.random()*0.5-0.25),
                piMod:Math.random()*2, mayaDigit:i%20, adjacency:adj});
        }
    }

    _geodesicDist(a,b){
        let s=0; const n=Math.min(a.length,b.length,4);
        for(let i=0;i<n;i++){const d=(a[i]-b[i])/20;s+=d*d;}
        const flat=Math.sqrt(s);
        return flat*(1+0.1*flat*flat/6);
    }

    step(){
        this.piTime=PiGeometry.normalize(this.piTime+this.dt*0.1);
        let totalCoh=0;
        for(let i=0;i<this.tokens.length;i++){
            const tok=this.tokens[i];
            const card=this.cards[tok.shardId%this.numCards];
            tok.phase=PiGeometry.rotate(tok.phase,this.dt*(card.piMod+this.piTime*0.1));
            const force=[0,0,0,0]; let totalInterf=0, shardPot=0;
            for(let j=0;j<this.tokens.length;j++){
                if(i===j) continue;
                const other=this.tokens[j];
                const otherCard=this.cards[other.shardId%this.numCards];
                const isAdj=other.shardId===tok.shardId||card.adjacency.includes(other.shardId);
                if(!isAdj) continue;
                const dist=this._geodesicDist(tok.pos,other.pos);
                const interf=PiGeometry.interference(tok.phase,other.phase,
                    card.amplitude.reduce((s,v)=>s+v,0)/4,
                    otherCard.amplitude.reduce((s,v)=>s+v,0)/4);
                const w=Math.exp(-dist*dist/(2*this.sigma*this.sigma))*Math.abs(interf);
                const d=other.pos.slice(0,4).map((v,k)=>v-tok.pos[k]);
                const dn=Math.hypot(...d)||1e-9;
                const sign=interf>0?1:-1;
                for(let k=0;k<4;k++) force[k]+=(d[k]/dn)*w*sign;
                totalInterf+=Math.abs(interf);
                if(interf>0.5&&other.shardId!==tok.shardId) shardPot+=interf;
            }
            if(totalInterf>0) for(let k=0;k<4;k++) force[k]/=totalInterf;
            for(let k=0;k<4;k++) force[k]+=card.gradient[k]||0;
            for(let k=0;k<4;k++){
                tok.vel[k]=tok.vel[k]*0.95+force[k]*this.dt;
                tok.pos[k]+=tok.vel[k]*this.dt;
            }
            const pm=Math.hypot(...tok.pos.slice(0,4));
            tok.lc=MayanCalendar.advance(tok.lc,Math.max(1,Math.floor(Math.abs(pm)*10)));
            if(tok.phase<this.dt*card.piMod||shardPot>1.0){
                const tgt=Math.floor(Math.abs(tok.pos[0])*20)%20;
                let found=tok.shardId;
                for(const adj of card.adjacency){
                    if(adj<this.numCards&&this.cards[adj].mayaDigit===tgt){found=adj;break;}
                }
                if(Math.random()<this.temperature)
                    found=card.adjacency[Math.floor(Math.random()*Math.min(8,card.adjacency.length))];
                if(found!==tok.shardId){this.shardTransitions++;tok.shardId=found;}
            }
            tok.coherence=tok.coherence*0.99+(totalInterf>0?totalInterf/100:0);
            totalCoh+=tok.coherence;
        }
        this.globalLC=MayanCalendar.advance(this.globalLC);
        this.tick++;
        const avg=totalCoh/this.numTokens;
        this.coherenceHistory.push(avg);
        return avg;
    }

    run(steps=500){
        for(let s=0;s<steps;s++) this.step();
        const avg=this.coherenceHistory.reduce((a,b)=>a+b,0)/this.coherenceHistory.length;
        return {finalCoherence:this.coherenceHistory.at(-1)??0,avgCoherence:avg,
                shardTransitions:this.shardTransitions,ticks:this.tick};
    }

    // Convert to optical node format for SVG3DComputeGraph.fromOpticalMesh()
    toOpticalNodes(){
        return this.tokens.map(t=>{
            const n=Math.hypot(...t.pos.slice(0,3))||1;
            const sh=new Float32Array(18);
            sh[0]=Math.cos(t.phase)*t.coherence;
            sh[1]=Math.sin(t.phase)*t.coherence;
            return {position:[t.pos[0]/n,t.pos[1]/n,t.pos[2]/n],
                    normal:[t.pos[0]/n,t.pos[1]/n,t.pos[2]/n],
                    uv:[0,0], sh, energy:t.coherence};
        });
    }

    metrics(){
        const lc=this.globalLC;
        const days=MayanCalendar.longCountToDays(lc.baktun,lc.katun,lc.tun,lc.uinal,lc.kin);
        const tz=MayanCalendar.tzolkin(days), haab=MayanCalendar.haab(days);
        const avg=this.tokens.reduce((s,t)=>s+t.coherence,0)/this.numTokens;
        return {coherence:avg,piTime:this.piTime,tick:this.tick,
                longCount:`${lc.baktun}.${lc.katun}.${lc.tun}.${lc.uinal}.${lc.kin}`,
                tzolkin:`${tz.number} ${['Imix','Ik','Akbal','Kan','Chicchan','Cimi','Manik','Lamat','Muluc','Oc','Chuen','Eb','Ben','Ix','Men','Cib','Caban','Eznab','Cauac','Ahau'][tz.dayIdx]}`,
                haab:`${haab.day} ${haab.month}`,
                shardTransitions:this.shardTransitions,
                activeTokens:this.tokens.filter(t=>t.coherence>0.1).length};
    }
}
