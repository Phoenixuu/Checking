#include "ORTYolov5.hpp"

#include <iostream>

#include "opencv2/dnn.hpp"


ORTYoloV5::ORTYoloV5(std::shared_ptr<ORTRunner> shpORTRunner, std::function<void (const std::vector<stObject_t>&)> fnCallback)
    : ORTModel(shpORTRunner), m_fnCallback(fnCallback)
{
    std::cout << "======gia tri ban dau=========" << std::endl;
    std::cout << umpInputTensorsShape.size() << std::endl;
    m_iWidthModel = umpInputTensorsShape[m_strInputName][3];
    std::cout << "====================1============" << std::endl;
    m_iHeightModel = umpInputTensorsShape[m_strInputName][2];
    std::cout << "====================2============" << std::endl;

    m_iOutputHeight = 5 + m_iNumClasses;
    std::cout << "=======Khoi tao========" << std::endl;
}



ORTYoloV5::~ORTYoloV5()
{
    
}

void ORTYoloV5::setLabels(std::string& strFileLabel)
{

}

void ORTYoloV5::setScoreThreshold(float fScoreThreshold)
{
    m_fScoreThreshold = fScoreThreshold;
}

void ORTYoloV5::setNMSThreshold(float fNMSThreshold)
{
    m_fNMSThreshold = fNMSThreshold;
}

void ORTYoloV5::preprocess(cv::Mat& mImage)
{
    std::cout << "===========preprocess=========" << std::endl;
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));
    cv::Mat mInput;
    cv::resize(mImage, mInput, cv::Size(m_iWidthModel, m_iHeightModel), 0, 0, cv::INTER_LINEAR);

    cv::cvtColor(mInput, mInput, cv::COLOR_BGR2RGB);
    cv::Mat mFloat;
    mInput.convertTo(mFloat, CV_32FC3, 1.f / 255.f);

    cv::Mat mChannels[3];
    cv::split(mFloat, mChannels);

    inputOrtValues.clear();
    for (auto& channel : mChannels)
    {
        std::vector<float> fVec(channel.begin<float>(), channel.end<float>());
        inputOrtValues.insert(inputOrtValues.end(), fVec.begin(), fVec.end());
    }
    std::cout << "=============done preprocess============" << std::endl;
}

void ORTYoloV5::run(cv::Mat& mImage, std::vector<stObject_t>& stObjects)
{
    preprocess(mImage);
    shpORTRunner->runModel(inputOrtValues, outputOrtValues);
    postprocess(stObjects);
}

void ORTYoloV5::postprocess()
{
    std::vector<float> fOutput = outputOrtValues[umpOutputTensors[m_strOutputName][0]];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;

    for (int i = 0; i < fOutput.size(); i += m_iOutputHeight)
    {
        float fScore = fOutput[i + 4];
        if (fScore >= m_fScoreThreshold)
        {
            std::vector<float> fScores;
            for (int j = 5; j < m_iOutputHeight; j++)
            {
                fScores.push_back(fOutput[i + j]);
            }
            auto maxScorePtr = std::max_element(fScores.begin(), fScores.end());
            float fScoreClass = *maxScorePtr;
            int iId = std::distance(fScores.begin(), maxScorePtr);

            float fScoreObject = fScore * fScoreClass;
            if (fScoreObject >= m_fScoreThreshold)
            {
                float x = fOutput[i + 0];
                float y = fOutput[i + 1];
                float w = fOutput[i + 2];
                float h = fOutput[i + 3];

                float x0 = std::clamp((x - 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y0 = std::clamp((y - 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);
                float x1 = std::clamp((x + 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y1 = std::clamp((y + 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);

                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;

                bboxes.push_back(bbox);
                scores.push_back(fScoreObject);
                classes.push_back(iId);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxesBatched(bboxes, scores, classes, m_fScoreThreshold, m_fNMSThreshold, indices);

    std::vector<stObject_t> stOutputs;
    int cnt = 0;
    for (auto& chosenIdx : indices)
    {
        stObject_t obj;
        obj.rfBox = bboxes[chosenIdx];
        obj.fScore = scores[chosenIdx];
        obj.iId = classes[chosenIdx];
        obj.strLabel = std::to_string(classes[chosenIdx]);
        stOutputs.push_back(obj);

        cnt += 1;
    }

    m_fnCallback(stOutputs);
}

void ORTYoloV5::postprocess(std::vector<stObject_t>& stObjects)
{
    std::cout << "vao dayyyyyyyyyyyyyyyyyy" << std::endl;
    stObjects.clear();

    std::vector<float> fOutput = outputOrtValues[umpOutputTensors[m_strOutputName][0]];
    for(int i = 0; i < 10; i++){
        std::cout << fOutput[6300*4+i] << std::endl;
    }
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;

    std::cout << fOutput.size() << std::endl;
    std::cout << m_iOutputHeight << std::endl;

    for (int i = 0; i < fOutput.size(); i += m_iOutputHeight)
    {
        float fScore = fOutput[i + 4];
        if (fScore >= m_fScoreThreshold)
        {
            std::vector<float> fScores;
            for (int j = 5; j < m_iOutputHeight; j++)
            {
                fScores.push_back(fOutput[i + j]);
            }
            auto maxScorePtr = std::max_element(fScores.begin(), fScores.end());
            float fScoreClass = *maxScorePtr;
            int iId = std::distance(fScores.begin(), maxScorePtr);

            float fScoreObject = fScore * fScoreClass;
            if (fScoreObject >= m_fScoreThreshold)
            {
                float x = fOutput[i + 0];
                float y = fOutput[i + 1];
                float w = fOutput[i + 2];
                float h = fOutput[i + 3];

                float x0 = std::clamp((x - 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y0 = std::clamp((y - 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);
                float x1 = std::clamp((x + 0.5f * w) * m_fRatioWidth, 0.f, (float)m_iInputWidth);
                float y1 = std::clamp((y + 0.5f * h) * m_fRatioHeight, 0.f, (float)m_iInputHeight);

                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;

                bboxes.push_back(bbox);
                scores.push_back(fScoreObject);
                classes.push_back(iId);
            }
        }




        // std::string outputDir = "results";
        // std::filesystem::create_directory(outputDir);  

        // std::string outputFile = outputDir + "/output.jpg";  S

        // cv::Mat img = mImage.clone(); // Giả sử mImage là ảnh đầu vào
        // for (const auto& bbox : bboxes)
        // {
        //     cv::rectangle(img, bbox, cv::Scalar(0, 255, 0), 2); // Vẽ bounding box
        // }

        // if (cv::imwrite(outputFile, img)) {
        //     std::cout << "Image saved to: " << outputFile << std::endl;
        // } else {
        //     std::cerr << "Failed to save image to: " << outputFile << std::endl;
        // }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxesBatched(bboxes, scores, classes, m_fScoreThreshold, m_fNMSThreshold, indices);

    int cnt = 0;
    for (auto& chosenIdx : indices)
    {
        stObject_t obj;
        obj.rfBox = bboxes[chosenIdx];
        obj.fScore = scores[chosenIdx];
        obj.iId = classes[chosenIdx];
        obj.strLabel = std::to_string(classes[chosenIdx]);
        stObjects.push_back(obj);

        std::cout << " a \n a \n a \n a \n a \n a \n a \n a" << std::endl;
        cv::imwrite("/home/dunggps/Face_Recognition_Vendor_Test/frvt-custom/11/Output/" + std::to_string(cnt) + ".jpg", bgr);
        cnt += 1;

        std::cout << cnt << std::endl;
    }
}
