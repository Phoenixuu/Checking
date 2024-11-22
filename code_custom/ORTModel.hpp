#ifndef ORTModel_hpp
#define ORTModel_hpp

#include <opencv2/core/core.hpp>

#include "ORTRunner.hpp"

class ORTModel
{
    public:
        ORTModel(std::shared_ptr<ORTRunner> shpRunner);
        ~ORTModel();

        void run(cv::Mat& mImage);

        std::vector<void*> buffers;

    protected:
        virtual void preprocess(cv::Mat& mImage) = 0;
        virtual void postprocess() = 0;

    protected:
        std::shared_ptr<ORTRunner> shpORTRunner;

        // std::unordered_map<std::string, std::vector<size_t>> umpIOTensors;
        // std::unordered_map<std::string, std::vector<int64_t>> umpIOTensorsShape;

        std::unordered_map<std::string, std::vector<size_t>> umpInputTensors;
        std::unordered_map<std::string, std::vector<int64_t>> umpInputTensorsShape;

        std::unordered_map<std::string, std::vector<size_t>> umpOutputTensors;


        std::vector<float> inputOrtValues;
        std::vector<std::vector<float>> outputOrtValues;

    };

#endif // ORTModel_hpp