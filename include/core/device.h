
#pragma once

namespace core {

enum class DeviceType {
    CPU,
    METAL
};

struct Device {
    DeviceType type = DeviceType::CPU;

    static Device cpu() {
        return {DeviceType::CPU};
    }

    static Device metal() {
        return {DeviceType::METAL};
    }
};

}
