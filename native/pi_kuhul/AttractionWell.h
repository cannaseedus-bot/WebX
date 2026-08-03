#pragma once

#include <DirectXMath.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace PiKuhul {

// Bounded radial field used to attract or repel bridge tokens.
class AttractionWell {
public:
    struct Parameters {
        DirectX::XMFLOAT3 position{0.0f, 0.0f, 0.0f};
        float strength = 2.0f;      // Positive attracts; negative repels.
        float radius = 5.0f;
        float falloffPower = 2.0f;

        static constexpr float StrengthMin = 0.1f;
        static constexpr float StrengthMax = 20.0f;
        static constexpr float RadiusMin = 1.0f;
        static constexpr float RadiusMax = 100.0f;
        static constexpr float FalloffMin = 0.5f;
        static constexpr float FalloffMax = 5.0f;

        bool Validate() const {
            return std::abs(strength) >= StrengthMin &&
                   std::abs(strength) <= StrengthMax &&
                   radius >= RadiusMin && radius <= RadiusMax &&
                   falloffPower >= FalloffMin && falloffPower <= FalloffMax;
        }
    };

    struct ForceSample {
        DirectX::XMFLOAT3 position{};
        DirectX::XMFLOAT3 force{};
        float magnitude = 0.0f;
        float normalizedDistance = 1.0f;
    };

    AttractionWell() = default;
    explicit AttractionWell(const Parameters& parameters) {
        SetParameters(parameters);
    }

    const Parameters& GetParameters() const { return parameters_; }

    bool SetParameters(const Parameters& parameters) {
        if (!parameters.Validate()) return false;
        parameters_ = parameters;
        return true;
    }

    void SetPosition(const DirectX::XMFLOAT3& position) {
        parameters_.position = position;
    }

    DirectX::XMFLOAT3 GetPosition() const { return parameters_.position; }

    bool SetStrength(float strength) {
        if (std::abs(strength) < Parameters::StrengthMin ||
            std::abs(strength) > Parameters::StrengthMax) return false;
        parameters_.strength = strength;
        return true;
    }

    float GetStrength() const { return parameters_.strength; }

    bool SetRadius(float radius) {
        if (radius < Parameters::RadiusMin || radius > Parameters::RadiusMax)
            return false;
        parameters_.radius = radius;
        return true;
    }

    float GetRadius() const { return parameters_.radius; }

    bool SetFalloffPower(float power) {
        if (power < Parameters::FalloffMin || power > Parameters::FalloffMax)
            return false;
        parameters_.falloffPower = power;
        return true;
    }

    float GetFalloffPower() const { return parameters_.falloffPower; }

    DirectX::XMFLOAT3 CalculateForce(
        const DirectX::XMFLOAT3& point) const {
        const float dx = point.x - parameters_.position.x;
        const float dy = point.y - parameters_.position.y;
        const float dz = point.z - parameters_.position.z;
        const float distanceSquared = dx * dx + dy * dy + dz * dz;
        const float distance = std::sqrt(distanceSquared);

        if (distance < 1.0e-8f || distance >= parameters_.radius)
            return {};

        const float t = distance / parameters_.radius;
        const float falloff =
            std::pow(std::max(0.0f, 1.0f - t), parameters_.falloffPower);
        const float scale = -parameters_.strength * falloff / distance;
        return {dx * scale, dy * scale, dz * scale};
    }

    ForceSample SampleField(const DirectX::XMFLOAT3& point) const {
        ForceSample sample;
        sample.position = point;
        sample.force = CalculateForce(point);
        sample.magnitude = std::sqrt(
            sample.force.x * sample.force.x +
            sample.force.y * sample.force.y +
            sample.force.z * sample.force.z);

        const float dx = point.x - parameters_.position.x;
        const float dy = point.y - parameters_.position.y;
        const float dz = point.z - parameters_.position.z;
        const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        sample.normalizedDistance =
            std::min(1.0f, distance / parameters_.radius);
        return sample;
    }

    template <typename Token>
    void ApplyToTokens(std::vector<Token>& tokens, float dt = 0.01f) const {
        if (dt < 0.0f) return;
        for (auto& token : tokens) {
            const DirectX::XMFLOAT3 force = CalculateForce(token.position);
            token.velocity.x += force.x * dt;
            token.velocity.y += force.y * dt;
            token.velocity.z += force.z * dt;
        }
    }

private:
    Parameters parameters_{};
};

} // namespace PiKuhul
