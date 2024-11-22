#include "ORTRunner.hpp"

ORTRunner::ORTRunner(const std::string& strModelPath)
{
    int iInputWidth = 32;
    int iInputHeight = 32;
    m_env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, strModelPath.c_str());
    m_sessionOptions = Ort::SessionOptions();
    m_session = new Ort::Session(m_env, strModelPath.c_str(), m_sessionOptions);

    size_t numInputNodes = m_session->GetInputCount();
    for (int i = 0; i < numInputNodes; i++)
    {
        Ort::AllocatedStringPtr inputNodeNameAllocated = m_session->GetInputNameAllocated(i, m_ortAllocator);
        const char* inputNodeName = std::move(inputNodeNameAllocated).release();
        m_inputNames.push_back(inputNodeName);

        m_inputTensorShape = {1, 3, iInputHeight, iInputWidth};
        size_t inputSize = vectorProduct(m_inputTensorShape);
        m_inputTensorSize = inputSize;

        m_umpInputTensors[inputNodeName] = {i, inputSize};
        m_umpInputTensorsShape[inputNodeName] = m_inputTensorShape;
    }

    size_t numOutputNodes = m_session->GetOutputCount();
    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::AllocatedStringPtr outputNodeNameAllocated = m_session->GetOutputNameAllocated(i, m_ortAllocator);
        const char* outputNodeName = std::move(outputNodeNameAllocated).release();
        m_outputNames.push_back(outputNodeName);

        m_umpOutputTensors[outputNodeName] = {i, 0};
    }
}

ORTRunner::~ORTRunner()
{
    delete m_session;
}

void ORTRunner::getInputInfo(std::unordered_map<std::string, std::vector<size_t>>& umpInputTensors, 
                            std::unordered_map<std::string, std::vector<int64_t>>& umpInputTensorsShape)
{
    umpInputTensors = m_umpInputTensors;
    umpInputTensorsShape = m_umpInputTensorsShape;
}

void ORTRunner::getOutputInfo(std::unordered_map<std::string, std::vector<size_t>>& umpOutputTensors)
{
    umpOutputTensors = m_umpOutputTensors;
}

void ORTRunner::runModel(std::vector<float>& inputOrtValues, std::vector<std::vector<float>>& outputOrtValues)
{
    std::vector<Ort::Value> ortInputTensors;
    Ort::MemoryInfo ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    outputOrtValues.clear();

    // Ort::MemoryInfo ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    size_t inputTensorSize = vectorProduct(m_inputTensorShape);

    ortInputTensors.push_back(
        Ort::Value::CreateTensor<float>(ortMemoryInfo, 
                                        inputOrtValues.data(), 
                                        inputTensorSize, 
                                        m_inputTensorShape.data(), 
                                        m_inputTensorShape.size()));
    
    m_outputTensors = m_session->Run(Ort::RunOptions{nullptr}, 
                                    m_inputNames.data(), 
                                    ortInputTensors.data(), 
                                    1, 
                                    m_outputNames.data(), 
                                    m_outputNames.size());

    for (auto& i : m_outputTensors)
    {
        auto* rawOutput = i.GetTensorData<float>();
        std::vector<int64_t> outputShape = i.GetTensorTypeAndShapeInfo().GetShape();
        size_t outputTensorSize = vectorProduct(outputShape);
        std::vector<float> outputTensor(rawOutput, rawOutput + outputTensorSize);
        outputOrtValues.push_back(outputTensor);
    }
}
