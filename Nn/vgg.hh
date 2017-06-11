#ifndef VGG_H
#define VGG_H

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

typedef std::vector<cv::Mat> Video;

class vgg
{
private:

    static const Core::ParameterString paramVideosDirectory_;

    std::string VideosDirectory_;

    static const Core::ParameterString paramVideoList_;
    std::string videoList_;

    void createTemporalFramesFromVideo(const std::string& filename);
    void readVideo(const std::string& filename, Video& result);
public:
    vgg();
    void prepareVideoForTemporalStream();
};

#endif // VGG_H
