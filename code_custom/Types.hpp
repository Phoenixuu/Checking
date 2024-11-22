#ifndef Types_hpp
#define Types_hpp

#include <opencv2/core/core.hpp>

/**
 * @brief structure of ...
*/
typedef struct stObject
{
    cv::Rect2f rfBox;
    float fScore;
    int iId;
    std::string strLabel;
} stObject_t;

/**
 * @brief 
*/
typedef struct stFaceObject
{
    cv::Rect2f rfBox;
    float fScore;
    std::vector<cv::Point2f> ptfLandmarks;
} stFaceObject_t;

/**
 * @brief 
*/
typedef struct stHumanAttribute {
    std::string strGender;
    float fScoreGender;
} stHumanAttribute_t;

#endif // Types_hpp