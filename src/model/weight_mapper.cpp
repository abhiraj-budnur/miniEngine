
#include "model/weight_mapper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace model
{

    WeightMapper::WeightMapper(
        const std::string &model_directory)
        : model_directory_(model_directory)
    {

        load_index();
    }

    void WeightMapper::load_index()
    {

        namespace fs = std::filesystem;

        fs::path index_path =
            fs::path(model_directory_) /
            "model.safetensors.index.json";

        if (fs::exists(index_path))
        {

            std::ifstream file(index_path);

            if (!file)
            {
                throw std::runtime_error(
                    "Cannot open model index");
            }

            json index;
            file >> index;

            if (!index.contains("weight_map"))
            {
                throw std::runtime_error(
                    "Invalid safetensors index");
            }

            for (auto it = index["weight_map"].begin();
                 it != index["weight_map"].end();
                 ++it)
            {

                tensor_to_shard_[it.key()] =
                    it.value().get<std::string>();
            }
        }
        else
        {

            // Support an unsharded checkpoint.
            fs::path single_file =
                fs::path(model_directory_) /
                "model.safetensors";

            if (!fs::exists(single_file))
            {
                throw std::runtime_error(
                    "No safetensors checkpoint or index found");
            }

            auto reader =
                std::make_unique<SafeTensorsReader>(
                    single_file.string());

            for (const auto &name : reader->tensor_names())
            {
                tensor_to_shard_[name] = "model.safetensors";
            }

            readers_["model.safetensors"] =
                std::move(reader);
        }

        if (tensor_to_shard_.empty())
        {
            throw std::runtime_error(
                "Checkpoint contains no indexed tensors");
        }
    }

    SafeTensorsReader &WeightMapper::get_reader(
        const std::string &shard)
    {

        auto it = readers_.find(shard);

        if (it != readers_.end())
        {
            return *it->second;
        }

        std::filesystem::path shard_path =
            std::filesystem::path(model_directory_) / shard;

        if (!std::filesystem::exists(shard_path))
        {
            throw std::runtime_error(
                "Missing checkpoint shard: " + shard);
        }

        auto reader =
            std::make_unique<SafeTensorsReader>(
                shard_path.string());

        SafeTensorsReader &reference = *reader;

        readers_[shard] = std::move(reader);

        return reference;
    }

    bool WeightMapper::contains(
        const std::string &tensor_name) const
    {

        return tensor_to_shard_.find(tensor_name) !=
               tensor_to_shard_.end();
    }

    core::Tensor WeightMapper::get(
        const std::string &tensor_name)
    {

        auto it = tensor_to_shard_.find(tensor_name);

        if (it == tensor_to_shard_.end())
        {
            throw std::runtime_error(
                "Checkpoint tensor missing: " + tensor_name);
        }

        auto &reader = get_reader(it->second);

        if (!reader.contains(tensor_name))
        {
            throw std::runtime_error(
                "Index and shard disagree for: " + tensor_name);
        }

        return reader.load_tensor(tensor_name);
    }

    std::vector<std::string>
    WeightMapper::tensor_names() const
    {

        std::vector<std::string> names;

        names.reserve(tensor_to_shard_.size());

        for (const auto &entry : tensor_to_shard_)
        {
            names.push_back(entry.first);
        }

        std::sort(names.begin(), names.end());

        return names;
    }

}
std::string model::WeightMapper::get_dtype(
    const std::string &tensor_name)
{
    auto it = tensor_to_shard_.find(tensor_name);

    if (it == tensor_to_shard_.end())
    {
        throw std::runtime_error(
            "Tensor not found: " + tensor_name);
    }

    return get_reader(it->second)
        .get_info(tensor_name)
        .dtype;
}