// Field System — JS port of v0.2.0-kuhul-directx-native π_field_v1 specs.
// Implements: AttractionWell, WindField, NavigationForce, ScrollInertiaField, FieldSystem.
//
// Body interface: { x, y, z, vx, vy, vz }
// Call FieldSystem.tick(bodies, dt) each frame.

// ------------------------------------------------------------------ //
// AttractionWell — attraction_well_spec.json
// ------------------------------------------------------------------ //

export class AttractionWell {
    constructor({
        position    = [0, 0, 0],
        strength    = 2.0,
        radius      = 5.0,
        falloffPower = 2.0,
    } = {}) {
        this.position    = position;
        this.strength    = strength;
        this.radius      = radius;
        this.falloffPower = falloffPower;
    }

    applyTo(body, dt) {
        const dx = this.position[0] - body.x;
        const dy = this.position[1] - body.y;
        const dz = this.position[2] - body.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist === 0 || dist > this.radius) return;
        const t     = 1 - dist / this.radius;
        const falloff = Math.pow(t, this.falloffPower);
        const force = (this.strength * falloff) / dist;
        body.vx += dx * force * dt;
        body.vy += dy * force * dt;
        body.vz += dz * force * dt;
    }
}

// ------------------------------------------------------------------ //
// WindField — wind_field_spec.json
// ------------------------------------------------------------------ //

export class WindField {
    constructor({
        direction = [1, 0, 0],
        strength  = 0.5,
    } = {}) {
        this.direction = direction;
        this.strength  = strength;
    }

    applyTo(body, dt) {
        body.vx += this.direction[0] * this.strength * dt;
        body.vy += this.direction[1] * this.strength * dt;
        body.vz += this.direction[2] * this.strength * dt;
    }
}

// ------------------------------------------------------------------ //
// NavigationForce — navigation_force_spec.json
// ------------------------------------------------------------------ //

export class NavigationForce {
    constructor({
        targetPosition  = [0, 0, 0],
        strength        = 2.0,
        arrivalRadius   = 1.0,
        maxSpeed        = 5.0,
        arrivalSlowdown = 2.0,
    } = {}) {
        this.targetPosition  = targetPosition;
        this.strength        = strength;
        this.arrivalRadius   = arrivalRadius;
        this.maxSpeed        = maxSpeed;
        this.arrivalSlowdown = arrivalSlowdown;
    }

    applyTo(body, dt) {
        const dx = this.targetPosition[0] - body.x;
        const dy = this.targetPosition[1] - body.y;
        const dz = this.targetPosition[2] - body.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.01) { body.vx = 0; body.vy = 0; body.vz = 0; return; }

        const slowScale = dist < this.arrivalSlowdown
            ? dist / this.arrivalSlowdown
            : 1.0;

        const nx = dx / dist, ny = dy / dist, nz = dz / dist;
        body.vx += nx * this.strength * slowScale * dt;
        body.vy += ny * this.strength * slowScale * dt;
        body.vz += nz * this.strength * slowScale * dt;

        const targetSpeed = this.maxSpeed * slowScale;
        const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy + body.vz * body.vz);
        if (speed > targetSpeed && speed > 0) {
            const s = targetSpeed / speed;
            body.vx *= s; body.vy *= s; body.vz *= s;
        }
    }
}

// ------------------------------------------------------------------ //
// ScrollInertiaField — scroll_inertia_field_spec.json
// ------------------------------------------------------------------ //

export class ScrollInertiaField {
    constructor({
        direction      = [0, 1, 0],
        initialSpeed   = 10.0,
        decayRate      = 2.0,
        strength       = 1.0,
        stopThreshold  = 0.1,
        bounceFactor   = 0.3,
        boundaryMin    = null,
        boundaryMax    = null,
    } = {}) {
        this.direction     = direction;
        this.speed         = initialSpeed;
        this.decayRate     = decayRate;
        this.strength      = strength;
        this.stopThreshold = stopThreshold;
        this.bounceFactor  = bounceFactor;
        this.boundaryMin   = boundaryMin;
        this.boundaryMax   = boundaryMax;
    }

    update(dt) {
        this.speed *= Math.exp(-this.decayRate * dt);
        if (this.speed < this.stopThreshold) this.speed = 0;
    }

    applyTo(body, dt) {
        if (this.speed === 0) return;
        const force = this.speed * this.strength;
        body.vx += this.direction[0] * force * dt;
        body.vy += this.direction[1] * force * dt;
        body.vz += this.direction[2] * force * dt;

        if (this.boundaryMin || this.boundaryMax) {
            const axes = ['x', 'y', 'z'];
            const vel  = ['vx', 'vy', 'vz'];
            for (let i = 0; i < 3; i++) {
                const lo = this.boundaryMin ? this.boundaryMin[i] : null;
                const hi = this.boundaryMax ? this.boundaryMax[i] : null;
                if (lo !== null && body[axes[i]] < lo) {
                    body[axes[i]] = lo;
                    body[vel[i]] *= -this.bounceFactor;
                }
                if (hi !== null && body[axes[i]] > hi) {
                    body[axes[i]] = hi;
                    body[vel[i]] *= -this.bounceFactor;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ //
// FieldSystem — manages a collection of fields and ticks bodies
// ------------------------------------------------------------------ //

export class FieldSystem {
    constructor({ damping = 0.97 } = {}) {
        this.fields  = [];
        this.damping = damping;
    }

    add(field) {
        this.fields.push(field);
        return this;
    }

    remove(field) {
        this.fields = this.fields.filter(f => f !== field);
        return this;
    }

    tick(bodies, dt) {
        for (const field of this.fields) {
            if (typeof field.update === 'function') field.update(dt);
        }
        for (const body of bodies) {
            for (const field of this.fields) {
                field.applyTo(body, dt);
            }
            body.x += body.vx * dt;
            body.y += body.vy * dt;
            body.z += body.vz * dt;
            body.vx *= this.damping;
            body.vy *= this.damping;
            body.vz *= this.damping;
        }
    }
}
