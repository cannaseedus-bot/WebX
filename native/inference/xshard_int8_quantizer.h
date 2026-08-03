#pragma once

#include <string>

class XShardInt8Quantizer {
public:
    bool convert(const std::string& inputPath, const std::string& outputPath);
    bool convertInt4(const std::string& inputPath,
                     const std::string& outputPath);
    const std::string& error() const { return error_; }

private:
    std::string error_;
};
