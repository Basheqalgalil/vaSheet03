#ifndef MYBATCHNORMALIZATIONLAYER_HH
#define MYBATCHNORMALIZATIONLAYER_HH

#include "MultiPortLayer.hh"

namespace Nn
{

class MyBatchNormalizationLayer : public MultiPortLayer
{
private:
    typedef MultiPortLayer Precursor;

    Vector gamma_;
    Vector beta_;
    Vector runningMean_;
    Vector runningVariance_;
    Vector saveMean_;
    Vector saveVariance_;
    Vector gammaDer_;
    Vector betaDer_;
    u32 nIterations_;
    u32 prevBatchSize_;


    void initializeParam(Vector& vector, const std::string& basePath,
            const std::string& suffix, const std::string& paramName, ParamInitialization initMethod);
    void save(Vector& vector, const std::string& basePath, const std::string& suffix, const std::string& paramName);
protected:
    void initializeParams(const std::string& basePath, const std::string& suffix);

public:
    MyBatchNormalizationLayer(const char* name);
    virtual void saveParams(const std::string& basePath, const std::string& suffix);
    virtual void forward(u32 port);
    virtual void backpropagate(u32 timeframe, u32 port);

    virtual void updateParams(f32 learningRate);
    virtual bool isBiasTrainable() const { return true; }

};
}
#endif // MYBATCHNORMALIZATIONLAYER_HH
