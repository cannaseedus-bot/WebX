// kuhul-es/runtime/src/browser.js
class KUHULRuntime {
  constructor() {
    this.π = new Map();  // Immutable bindings
    this.τ = new Map();  // Temporal bindings
    this.τHistory = new Map();  // History for replay
    this.glyphQueue = [];
    this.frame = 0;
    this.hashChain = [];
    this.world = {
      bodies: [],
      fields: [],
      active: true
    };
    
    this.eventHandlers = new Map();
    this.cssVER = new CSSVER();
    
    // Built-in glyph implementations
    this.glyphImplementations = {
      Sek: this.executeSek.bind(this),
      Pop: this.executePop.bind(this),
      Wo: this.executeWo.bind(this),
      'Ch\'en': this.executeChen.bind(this),
      Yax: this.executeYax.bind(this),
      Xul: this.executeXul.bind(this)
    };
  }
  
  // Event system
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }
  
  emit(event, data) {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => handler(data));
  }
  
  // Execute KUHUL-ES source
  async execute(source) {
    try {
      // Parse π and τ bindings
      this.parseBindings(source);
      
      // Parse and queue glyph calls
      this.parseGlyphCalls(source);
      
      // Execute the queue
      await this.executeQueue();
      
      this.emit('complete', { 
        frame: this.frame, 
        hashChain: this.hashChain,
        πBindings: this.π.size,
        τBindings: this.τ.size
      });
      
    } catch (error) {
      this.emit('error', error);
      console.error('KUHUL execution error:', error);
    }
  }
  
  parseBindings(source) {
    // Parse π bindings: π name = value;
    const πMatches = source.matchAll(/π\s+(\w+)\s*=\s*([^;]+)/g);
    for (const match of πMatches) {
      const name = match[1];
      const value = this.evaluateExpression(match[2]);
      this.π.set(name, Object.freeze(value));
    }
    
    // Parse τ bindings: τ name = value;
    const τMatches = source.matchAll(/τ\s+(\w+)\s*=\s*([^;]+)/g);
    for (const match of τMatches) {
      const name = match[1];
      const value = this.evaluateExpression(match[2]);
      this.τ.set(name, value);
      this.τHistory.set(name, []);
    }
    
    this.emit('stat_update', {
      piBindings: this.π.size,
      tauBindings: this.τ.size
    });
  }
  
  parseGlyphCalls(source) {
    // Parse glyph calls: yield* GlyphName(...);
    const glyphRegex = /yield\*\s*(Sek|Pop|Wo|Ch'en|Yax|Xul)\(([^)]*)\)/g;
    let match;
    
    while ((match = glyphRegex.exec(source)) !== null) {
      const glyph = match[1];
      const argsText = match[2];
      const args = this.parseArguments(argsText);
      
      this.glyphQueue.push({
        glyph,
        args,
        position: match.index,
        source: match[0]
      });
    }
  }
  
  evaluateExpression(expr) {
    // Safe evaluation of simple expressions
    expr = expr.trim();
    
    // Array literal
    if (expr.startsWith('[') && expr.endsWith(']')) {
      try {
        return JSON.parse(expr);
      } catch {
        // Fallback for complex arrays
        return expr;
      }
    }
    
    // Object literal
    if (expr.startsWith('{') && expr.endsWith('}')) {
      try {
        return JSON.parse(expr);
      } catch {
        return expr;
      }
    }
    
    // Number
    if (!isNaN(parseFloat(expr))) {
      return parseFloat(expr);
    }
    
    // String (remove quotes)
    if ((expr.startsWith("'") && expr.endsWith("'")) || 
        (expr.startsWith('"') && expr.endsWith('"'))) {
      return expr.slice(1, -1);
    }
    
    // Boolean
    if (expr === 'true') return true;
    if (expr === 'false') return false;
    
    // Variable reference
    if (this.π.has(expr)) return this.π.get(expr);
    if (this.τ.has(expr)) return this.τ.get(expr);
    
    return expr;
  }
  
  parseArguments(argsText) {
    const args = [];
    let current = '';
    let inString = false;
    let stringChar = '';
    let depth = 0;
    
    for (let i = 0; i < argsText.length; i++) {
      const char = argsText[i];
      
      if (!inString && (char === "'" || char === '"')) {
        inString = true;
        stringChar = char;
        current += char;
      } else if (inString && char === stringChar && argsText[i-1] !== '\\') {
        inString = false;
        current += char;
      } else if (!inString && char === '[') {
        depth++;
        current += char;
      } else if (!inString && char === ']') {
        depth--;
        current += char;
      } else if (!inString && char === '{') {
        depth++;
        current += char;
      } else if (!inString && char === '}') {
        depth--;
        current += char;
      } else if (!inString && depth === 0 && char === ',') {
        args.push(this.evaluateExpression(current.trim()));
        current = '';
      } else {
        current += char;
      }
    }
    
    if (current.trim()) {
      args.push(this.evaluateExpression(current.trim()));
    }
    
    return args;
  }
  
  async executeQueue() {
    for (const call of this.glyphQueue) {
      const result = await this.executeGlyph(call.glyph, call.args);
      
      // Update τ bindings if specified in result
      if (result && result.updateTau) {
        for (const [key, value] of Object.entries(result.updateTau)) {
          if (this.τ.has(key)) {
            const history = this.τHistory.get(key);
            history.push({
              frame: this.frame,
              value: value,
              hash: this.hashValue(value)
            });
            this.τ.set(key, value);
          }
        }
      }
      
      // Hash the execution for determinism
      const stateHash = this.hashState({
        frame: this.frame,
        glyph: call.glyph,
        args: call.args,
        result: result,
        π: Object.fromEntries(this.π),
        τ: Object.fromEntries(this.τ)
      });
      
      this.hashChain.push(stateHash);
      this.emit('hash', { frame: this.frame, hash: stateHash });
      
      this.frame++;
      this.emit('frame_update', this.frame);
      
      // Small delay for animation
      await new Promise(resolve => setTimeout(resolve, 16));
    }
  }
  
  async executeGlyph(glyph, args) {
    const implementation = this.glyphImplementations[glyph];
    if (!implementation) {
      console.warn(`Unknown glyph: ${glyph}`);
      return null;
    }
    
    return await implementation(...args);
  }
  
  async executeSek(operation, ...args) {
    this.emit('log', `[Sek] ${operation}: ${JSON.stringify(args)}`);
    
    switch (operation) {
      case 'log':
        console.log('[KUHUL]', ...args);
        return { message: args.join(' ') };
        
      case 'create_body':
        const body = args[0];
        this.world.bodies.push(body);
        this.emit('body_created', body);
        return { body };
        
      case 'update_physics':
        // Simple physics simulation
        const dt = args[0] || 0.016;
        this.world.bodies.forEach(body => {
          // Gravity
          body.vy += 9.81 * dt * 0.1;
          
          // Update position
          body.x += body.vx * dt;
          body.y += body.vy * dt;
          
          // Bounce off walls
          if (body.x < 0 || body.x > 800) body.vx *= -0.9;
          if (body.y < 0 || body.y > 400) body.vy *= -0.9;
        });
        return { dt, bodyCount: this.world.bodies.length };
        
      case 'render_frame':
        this.emit('render', this.world.bodies);
        return { rendered: true };
        
      case 'add_field':
        const field = args[0];
        this.world.fields.push(field);
        return { field };
        
      case 'start_physics':
        const fps = args[1] || 60;
        this.world.active = true;
        return { started: true, fps };
        
      default:
        return { operation, args, note: 'Not implemented' };
    }
  }
  
  async executePop(value) {
    this.emit('log', `[Pop] ${value}`);
    return { value };
  }
  
  async executeWo(operation, ...args) {
    this.emit('log', `[Wo] ${operation}: ${JSON.stringify(args)}`);
    
    if (operation === 'set') {
      const [key, value] = args;
      // Would set a τ binding
      return { updateTau: { [key]: value } };
    }
    
    return { operation, args };
  }
  
  async executeChen(source, ...args) {
    this.emit('log', `[Ch'en] Reading from ${source}: ${JSON.stringify(args)}`);
    
    // Simulate reading data
    return { source, data: 'sample data', timestamp: Date.now() };
  }
  
  async executeYax(condition, value) {
    this.emit('log', `[Yax] Condition: ${condition}, Value: ${value}`);
    
    // Simple condition evaluation
    const result = condition ? value : null;
    return { condition, value: result };
  }
  
  async executeXul() {
    this.emit('log', '[Xul] Stopping execution');
    this.world.active = false;
    return { stopped: true };
  }
  
  hashState(state) {
    const str = JSON.stringify(state);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return hash.toString(16);
  }
  
  hashValue(value) {
    return this.hashState({ value });
  }
  
  // Replay from a specific frame
  replayFrom(frame) {
    // Would restore state from hash chain
    this.emit('log', `[Replay] From frame ${frame}`);
    return { replaying: true, frame };
  }
}

// CSS-VER Integration
class CSSVER {
  constructor() {
    this.agents = new Map();
    this.cssVariables = new Map();
  }
  
  createAgent(element, bodyId) {
    const agent = {
      element,
      bodyId,
      cssVars: new Map([
        ['--π-x', '0px'],
        ['--π-y', '0px'],
        ['--π-scale', '1'],
        ['--π-rotation', '0deg']
      ])
    };
    
    this.agents.set(bodyId, agent);
    this.updateElement(agent);
    
    return agent;
  }
  
  updateFromPhysics(body) {
    const agent = this.agents.get(body.id);
    if (agent) {
      agent.cssVars.set('--π-x', `${body.x}px`);
      agent.cssVars.set('--π-y', `${body.y}px`);
      this.updateElement(agent);
    }
  }
  
  updateElement(agent) {
    for (const [prop, value] of agent.cssVars) {
      agent.element.style.setProperty(prop, value);
    }
  }
}

// Make it available globally
if (typeof window !== 'undefined') {
  window.KUHULRuntime = KUHULRuntime;
  window.CSSVER = CSSVER;
}

export { KUHULRuntime, CSSVER };