#ifndef ORTRunner_hpp
#define ORTRunner_hpp

#include <string>
#include <numeric>

#include <onnxruntime_cxx_api.h>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

class ORTRunner
{
    public:
        ORTRunner(const std::string& strModelPath);
        ~ORTRunner();

        void getInputInfo(std::unordered_map<std::string, std::vector<size_t>>& umpInputTensors, 
                    std::unordered_map<std::string, std::vector<int64_t>>& umpInputTensorsShape);

        void getOutputInfo(std::unordered_map<std::string, std::vector<size_t>>& umpOutputTensors);

        std::vector<Ort::Value> m_outputTensors;
        void runModel(std::vector<float>& inputOrtValues, std::vector<std::vector<float>>& outputOrtValues);

    private:
        Ort::Env m_env{nullptr};
        Ort::SessionOptions m_sessionOptions{nullptr};
        Ort::Session* m_session;

        Ort::AllocatorWithDefaultOptions m_ortAllocator;
        // Ort::MemoryInfo m_ortMemoryInfo;

        std::shared_ptr<char> _inputName, _outputName;
        std::vector<const char*> m_inputNames;
        std::vector<const char*> m_outputNames;

        std::vector<int64_t> m_inputTensorShape;
        size_t m_inputTensorSize;

        // std::unordered_map<std::string, std::vector<size_t>> m_umpIOTensors;
        // std::unordered_map<std::string, std::vector<int64_t>> m_umpIOTensorsShape;

        std::unordered_map<std::string, std::vector<size_t>> m_umpInputTensors;
        std::unordered_map<std::string, std::vector<int64_t>> m_umpInputTensorsShape;

        std::unordered_map<std::string, std::vector<size_t>> m_umpOutputTensors;
};

#endif // ORTRunner_hpp