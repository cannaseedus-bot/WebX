// kuhul-es/runtime/src/node.js
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class KUHULRuntimeNode {
  constructor() {
    this.π = new Map();
    this.τ = new Map();
    this.τHistory = new Map();
    this.glyphQueue = [];
    this.frame = 0;
    this.hashChain = [];
    this.world = {
      bodies: [],
      fields: [],
      active: true
    };
    
    this.eventHandlers = new Map();
    this.outputStream = process.stdout;
  }
  
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }
  
  emit(event, data) {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => {
      if (typeof handler === 'function') {
        handler(data);
      }
    });
  }
  
  async execute(source) {
    try {
      // Parse bindings
      this.parseBindings(source);
      
      // Parse glyph calls
      this.parseGlyphCalls(source);
      
      // Execute
      await this.executeQueue();
      
      this.log(`✓ Execution complete. Frame: ${this.frame}, Hash chain length: ${this.hashChain.length}`);
      
    } catch (error) {
      this.log(`✗ Execution error: ${error.message}`);
      throw error;
    }
  }
  
  async executeFile(filename) {
    const source = fs.readFileSync(filename, 'utf-8');
    return await this.execute(source);
  }
  
  parseBindings(source) {
    // Parse π bindings
    const πMatches = [...source.matchAll(/π\s+(\w+)\s*=\s*([^;]+)/g)];
    πMatches.forEach(match => {
      const name = match[1];
      const value = this.evaluateExpression(match[2]);
      this.π.set(name, Object.freeze(value));
    });
    
    // Parse τ bindings
    const τMatches = [...source.matchAll(/τ\s+(\w+)\s*=\s*([^;]+)/g)];
    τMatches.forEach(match => {
      const name = match[1];
      const value = this.evaluateExpression(match[2]);
      this.τ.set(name, value);
      this.τHistory.set(name, []);
    });
    
    this.log(`Parsed ${this.π.size} π-bindings and ${this.τ.size} τ-bindings`);
  }
  
  parseGlyphCalls(source) {
    const glyphRegex = /yield\*\s*(Sek|Pop|Wo|Ch'en|Yax|Xul)\(([^)]*)\)/g;
    let match;
    
    while ((match = glyphRegex.exec(source)) !== null) {
      const glyph = match[1];
      const argsText = match[2];
      const args = this.parseArguments(argsText);
      
      this.glyphQueue.push({
        glyph,
        args,
        position: match.index
      });
    }
    
    this.log(`Queued ${this.glyphQueue.length} glyph calls`);
  }
  
  async executeQueue() {
    for (const call of this.glyphQueue) {
      const result = await this.executeGlyph(call.glyph, call.args);
      
      // Update τ bindings
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
      
      // Hash state
      const stateHash = this.hashState({
        frame: this.frame,
        glyph: call.glyph,
        args: call.args,
        result: result
      });
      
      this.hashChain.push(stateHash);
      
      this.frame++;
      
      // Small delay
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
  
  async executeGlyph(glyph, args) {
    switch (glyph) {
      case 'Sek':
        return await this.executeSek(...args);
      case 'Pop':
        return await this.executePop(...args);
      case 'Wo':
        return await this.executeWo(...args);
      case 'Ch\'en':
        return await this.executeChen(...args);
      case 'Yax':
        return await this.executeYax(...args);
      case 'Xul':
        return await this.executeXul(...args);
      default:
        this.log(`Unknown glyph: ${glyph}`);
        return null;
    }
  }
  
  async executeSek(operation, ...args) {
    this.log(`[Sek] ${operation}: ${JSON.stringify(args)}`);
    
    switch (operation) {
      case 'log':
        this.outputStream.write(`[LOG] ${args.join(' ')}\n`);
        return { logged: true };
        
      case 'write_file':
        const [filename, content] = args;
        fs.writeFileSync(filename, content);
        return { file: filename, written: true };
        
      case 'read_file':
        const [filepath] = args;
        const content2 = fs.readFileSync(filepath, 'utf-8');
        return { file: filepath, content: content2 };
        
      case 'hash_file':
        const [filepath2] = args;
        const hash = crypto.createHash('sha256');
        const fileContent = fs.readFileSync(filepath2);
        hash.update(fileContent);
        return { file: filepath2, hash: hash.digest('hex') };
        
      default:
        return { operation, args };
    }
  }
  
  async executePop(value) {
    this.log(`[Pop] ${value}`);
    return { value };
  }
  
  async executeWo(operation, ...args) {
    this.log(`[Wo] ${operation}: ${JSON.stringify(args)}`);
    return { operation, args };
  }
  
  async executeChen(source, ...args) {
    this.log(`[Ch'en] ${source}: ${JSON.stringify(args)}`);
    return { source, args, timestamp: Date.now() };
  }
  
  async executeYax(condition, value) {
    this.log(`[Yax] Condition: ${condition}, Value: ${value}`);
    const result = condition ? value : null;
    return { condition, value: result };
  }
  
  async executeXul() {
    this.log('[Xul] Stopping execution');
    this.world.active = false;
    return { stopped: true };
  }
  
  evaluateExpression(expr) {
    expr = expr.trim();
    
    // Try JSON parsing first
    if ((expr.startsWith('[') && expr.endsWith(']')) || 
        (expr.startsWith('{') && expr.endsWith('}'))) {
      try {
        return JSON.parse(expr);
      } catch {
        // Continue with other methods
      }
    }
    
    // Number
    if (!isNaN(parseFloat(expr))) {
      return parseFloat(expr);
    }
    
    // String
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
  
  hashState(state) {
    const str = JSON.stringify(state);
    return crypto.createHash('sha256').update(str).digest('hex');
  }
  
  hashValue(value) {
    return this.hashState({ value });
  }
  
  log(message) {
    this.outputStream.write(`[KUHUL] ${message}\n`);
  }
  
  // Save state for replay
  saveState(filename) {
    const state = {
      π: Object.fromEntries(this.π),
      τ: Object.fromEntries(this.τ),
      τHistory: Object.fromEntries(this.τHistory),
      frame: this.frame,
      hashChain: this.hashChain,
      world: this.world
    };
    
    fs.writeFileSync(filename, JSON.stringify(state, null, 2));
    this.log(`State saved to ${filename}`);
  }
  
  // Load state for replay
  loadState(filename) {
    const state = JSON.parse(fs.readFileSync(filename, 'utf-8'));
    
    this.π = new Map(Object.entries(state.π));
    this.τ = new Map(Object.entries(state.τ));
    this.τHistory = new Map(Object.entries(state.τHistory));
    this.frame = state.frame;
    this.hashChain = state.hashChain;
    this.world = state.world;
    
    this.log(`State loaded from ${filename}. Frame: ${this.frame}`);
  }
}

module.exports = { KUHULRuntimeNode };