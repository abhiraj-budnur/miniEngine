
#include "model/safetensors_reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <cstdint>

using json = nlohmann::json;
namespace
{

    float bf16_to_float(uint16_t value)
    {
        uint32_t bits =
            static_cast<uint32_t>(value) << 16;

        float result;

        std::memcpy(
            &result,
            &bits,
            sizeof(float));

        return result;
    }

    float fp16_to_float(uint16_t value)
    {
        uint32_t sign =
            (static_cast<uint32_t>(value & 0x8000)) << 16;

        uint32_t exponent = (value >> 10) & 0x1F;
        uint32_t mantissa = value & 0x03FF;

        uint32_t bits = 0;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                int exp = -14;

                while ((mantissa & 0x0400) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }

                mantissa &= 0x03FF;

                bits = sign |
                       (static_cast<uint32_t>(exp + 127) << 23) |
                       (mantissa << 13);
            }
        }
        else if (exponent == 31)
        {
            bits = sign |
                   0x7F800000 |
                   (mantissa << 13);
        }
        else
        {
            bits = sign |
                   ((exponent - 15 + 127) << 23) |
                   (mantissa << 13);
        }

        float result;

        std::memcpy(
            &result,
            &bits,
            sizeof(float));

        return result;
    }

}
namespace model
{

    SafeTensorsReader::SafeTensorsReader(
        const std::string &filepath)
        : filepath_(filepath)
    {

        file_.open(
            filepath_,
            std::ios::binary | std::ios::ate);

        if (!file_)
        {
            throw std::runtime_error(
                "Cannot open safetensors file: " + filepath_);
        }

        auto end = file_.tellg();

        if (end < 0)
        {
            throw std::runtime_error(
                "Cannot determine safetensors file size");
        }

        file_size_ = static_cast<uint64_t>(end);

        read_header();
    }

    void SafeTensorsReader::read_header()
    {

        if (file_size_ < 8)
        {
            throw std::runtime_error(
                "Invalid safetensors file");
        }

        file_.seekg(0);

        uint64_t header_size = 0;

        file_.read(
            reinterpret_cast<char *>(&header_size),
            sizeof(header_size));

        if (!file_ ||
            header_size == 0 ||
            header_size > 100ULL * 1024 * 1024 ||
            header_size > file_size_ - 8)
        {

            throw std::runtime_error(
                "Invalid safetensors header size");
        }

        std::string header(
            static_cast<size_t>(header_size),
            '\0');

        file_.read(
            header.data(),
            static_cast<std::streamsize>(header_size));

        if (!file_)
        {
            throw std::runtime_error(
                "Failed reading safetensors header");
        }

        data_start_ = 8 + header_size;

        json metadata;

        try
        {
            metadata = json::parse(header);
        }
        catch (...)
        {
            throw std::runtime_error(
                "Malformed safetensors JSON header");
        }

        uint64_t data_size = file_size_ - data_start_;

        for (auto it = metadata.begin();
             it != metadata.end();
             ++it)
        {

            if (it.key() == "__metadata__")
            {
                continue;
            }

            const json &entry = it.value();

            TensorInfo info;

            info.name = it.key();

            info.dtype = entry.at("dtype").get<std::string>();

            info.shape =
                entry.at("shape").get<std::vector<size_t>>();

            auto offsets =
                entry.at("data_offsets").get<std::vector<uint64_t>>();

            if (offsets.size() != 2 ||
                offsets[0] > offsets[1] ||
                offsets[1] > data_size)
            {

                throw std::runtime_error(
                    "Invalid tensor offsets: " + info.name);
            }

            info.data_begin = offsets[0];
            info.data_end = offsets[1];

            uint64_t elements = 1;

            for (size_t dimension : info.shape)
            {

                if (dimension == 0)
                {
                    elements = 0;
                    break;
                }

                if (elements >
                    std::numeric_limits<uint64_t>::max() / dimension)
                {

                    throw std::runtime_error(
                        "Tensor shape overflow: " + info.name);
                }

                elements *= dimension;
            }

            uint64_t element_size = 0;

            if (info.dtype == "F32")
            {
                element_size = 4;
            }
            else if (info.dtype == "F16" ||
                     info.dtype == "BF16")
            {
                element_size = 2;
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported tensor dtype: " + info.dtype);
            }

            if (elements >
                std::numeric_limits<uint64_t>::max() / element_size)
            {

                throw std::runtime_error(
                    "Tensor byte size overflow");
            }

            uint64_t expected_bytes =
                elements * element_size;

            if (info.data_end - info.data_begin !=
                expected_bytes)
            {

                throw std::runtime_error(
                    "Tensor byte size mismatch: " + info.name);
            }

            if (!tensors_.emplace(info.name, info).second)
            {
                throw std::runtime_error(
                    "Duplicate tensor name: " + info.name);
            }
        }

        // Check that tensor data ranges do not overlap.
        std::vector<std::pair<uint64_t, uint64_t>> ranges;

        for (const auto &entry : tensors_)
        {
            ranges.emplace_back(
                entry.second.data_begin,
                entry.second.data_end);
        }

        std::sort(ranges.begin(), ranges.end());

        for (size_t i = 1; i < ranges.size(); ++i)
        {

            if (ranges[i].first < ranges[i - 1].second)
            {
                throw std::runtime_error(
                    "Overlapping safetensors data ranges");
            }
        }
    }

    bool SafeTensorsReader::contains(
        const std::string &name) const
    {

        return tensors_.find(name) != tensors_.end();
    }

    std::vector<std::string>
    SafeTensorsReader::tensor_names() const
    {

        std::vector<std::string> names;

        names.reserve(tensors_.size());

        for (const auto &entry : tensors_)
        {
            names.push_back(entry.first);
        }

        std::sort(names.begin(), names.end());

        return names;
    }

    const TensorInfo &SafeTensorsReader::get_info(
        const std::string &name) const
    {

        auto it = tensors_.find(name);

        if (it == tensors_.end())
        {
            throw std::runtime_error(
                "Tensor not found: " + name);
        }

        return it->second;
    }

    core::Tensor SafeTensorsReader::load_tensor(
        const std::string &name)
    {
        const TensorInfo &info = get_info(name);

        if (info.dtype != "F32" &&
            info.dtype != "F16" &&
            info.dtype != "BF16")
        {
            throw std::runtime_error(
                "Unsupported tensor dtype: " + info.dtype);
        }

        core::Tensor tensor(
            info.shape,
            core::DType::FP32);

        uint64_t bytes =
            info.data_end - info.data_begin;

        if (bytes == 0)
        {
            return tensor;
        }

        file_.clear();

        file_.seekg(
            static_cast<std::streamoff>(
                data_start_ + info.data_begin));

        if (!file_)
        {
            throw std::runtime_error(
                "Failed seeking to tensor: " + name);
        }

        if (info.dtype == "F32")
        {
            file_.read(
                reinterpret_cast<char *>(tensor.data()),
                static_cast<std::streamsize>(bytes));

            if (!file_)
            {
                throw std::runtime_error(
                    "Failed reading FP32 tensor: " + name);
            }
        }
        else
        {
            const size_t elements = tensor.numel();

            std::vector<uint16_t> source(elements);

            file_.read(
                reinterpret_cast<char *>(source.data()),
                static_cast<std::streamsize>(bytes));

            if (!file_)
            {
                throw std::runtime_error(
                    "Failed reading reduced precision tensor: " + name);
            }

            float *destination = tensor.data();

            for (size_t i = 0; i < elements; i++)
            {
                if (info.dtype == "BF16")
                {
                    destination[i] = bf16_to_float(source[i]);
                }
                else
                {
                    destination[i] = fp16_to_float(source[i]);
                }
            }
        }

        return tensor;
    }
}