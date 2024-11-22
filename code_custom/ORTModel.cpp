#include "ORTModel.hpp"

ORTModel::ORTModel(std::shared_ptr<ORTRunner> shpRunner) 
    : shpORTRunner(shpRunner)
{
    shpORTRunner->getInputInfo(umpInputTensors, umpInputTensorsShape);

    shpORTRunner->getOutputInfo(umpOutputTensors);
}

ORTModel::~ORTModel()
{
    for (int i = 0; i < this->buffers.size(); i++)
    {
        if (this->buffers[i] != nullptr)
        {
            free(this->buffers[i]);
        }
    }
    this->buffers.clear();
}

void ORTModel::run(cv::Mat& mImage)
{
    preprocess(mImage);
    shpORTRunner->runModel(inputOrtValues, outputOrtValues);
    postprocess();
}
