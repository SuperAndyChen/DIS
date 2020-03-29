#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "algorithm.h"
using namespace std;
using namespace cv;

class  MY_ORB
{
public:
    MY_ORB(float scaleFactor_, int nlevels_, int firstLevel_, int border_, int patchSize_, int fastThreshold_, 
        bool specifyNumberEachLevel_,bool useFastScore_, bool useHarrisScore_, int harrisBlockSize_, float harris_k_,
        int gaussianSize_, double gaussianSigma_) :
        scaleFactor(scaleFactor_), nlevels(nlevels_), firstLevel(firstLevel_),border(border_), patchSize(patchSize_), fastThreshold(fastThreshold_),
        specifyNumberEachLevel(specifyNumberEachLevel_), useFastScore(useFastScore_), useHarrisScore(useHarrisScore_), harrisBlockSize(harrisBlockSize_), harris_k(harris_k_),
        gaussianSize(gaussianSize_), gaussianSigma(gaussianSigma_)
    {}

    void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
    double getScaleFactor() { return scaleFactor; }

    void setNLevels(int nlevels_) { nlevels = nlevels_; }
    int getNLevels() { return nlevels; }

    void setFirstLevels(int firstLevel_) { firstLevel = firstLevel_; }
    int getFirstLevels() { return firstLevel; }

    void setBorder(int border_) { border = border_; }
    int getBorder() { return border; }

    void setPatchSize(int patchSize_) { patchSize = patchSize_; }
    int getPatchSize() { return patchSize; }

    void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
    int getFastThreshold() { return fastThreshold; }

    void setSpecifyNumberEachLevel(bool specifyNumberEachLevel_) { specifyNumberEachLevel = specifyNumberEachLevel_; }
    bool getSpecifyNumberEachLevel() { return specifyNumberEachLevel; }

    void setUseFastScore(bool useFastScore_) { useFastScore = useFastScore_; }
    bool getUseFastScore() { return useFastScore; }

    void setUseHarrisScore(bool useHarrisScore_) { useHarrisScore = useHarrisScore_; }
    bool getUseHarrisScore() { return useHarrisScore; }

    void setHarrisBlockSize(int harrisBlockSize_) { harrisBlockSize = harrisBlockSize_; }
    int getHarrisBlockSize() { return harrisBlockSize; }

    void setHarris_k(float harris_k_) { harris_k = harris_k_; }
    float getHarris_k() { return harris_k; }

    void setGaussianSize(int gaussianSize_) { gaussianSize = gaussianSize_; }
    int getGaussianSize() { return gaussianSize; }

    void setGaussianSigma(double gaussianSigma_) { gaussianSigma = gaussianSigma_; }
    double getGaussianSigma() { return gaussianSigma; }

    void detectAndCompute(Mat image, vector<KeyPoint>& keypoints, Mat& descriptors);


protected:

    double scaleFactor;
    int nlevels;
    int firstLevel;
    int border;
    int patchSize;
    int fastThreshold;
    bool specifyNumberEachLevel;
    bool useFastScore;
    bool useHarrisScore;
    int harrisBlockSize;
    float harris_k;
    int gaussianSize;
    double gaussianSigma;
};

static inline float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}

static void HarrisResponses(vector<Mat>& imagePyramid, vector<KeyPoint>& keypoints, int blockSize, float harris_k) {
    size_t ptidx, ptsize = keypoints.size();
    int r = blockSize / 2;
    float scale = 1.f / ((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;
    
    for (ptidx = 0; ptidx < ptsize; ptidx++) {
        
        int x0 = cvRound(keypoints[ptidx].pt.x);
        int y0 = cvRound(keypoints[ptidx].pt.y);
        
        int z = keypoints[ptidx].octave;
        int step = imagePyramid[z].size().width;
        
        const uchar* startingPoint = &imagePyramid[z].at<uchar>(y0 - r, x0 - r);
        int a = 0, b = 0, c = 0;
        for (size_t row = 0; row < blockSize; row++) {
            for (size_t col = 0; col < blockSize; col++) {
                const uchar* ptr = startingPoint + row * step + col;
                int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
                int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
                a += Ix * Ix;
                b += Iy * Iy;
                c += Ix * Iy;
            }
        }
        keypoints[ptidx].response = ((float)a * b - (float)c * c -
            harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
    }

}

void makeImagePyramid(Mat image, vector<Mat>& imagePyramid, vector<float>&layerScale, double scaleFactor, int nlevels, int firstLevel) {
    int level, nLevels = nlevels;
    Mat prevImg = image, currImg;
    for (level = 0; level < nlevels; level++) {
        float scale = getScale(level, firstLevel, scaleFactor);
        layerScale[level] = scale;
        Size sz(cvRound(image.cols / scale), cvRound(image.rows / scale));
        resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR_EXACT);
        imagePyramid.push_back(currImg);
        if (level > firstLevel)
            prevImg = currImg;
    }
}

static void getUMax(vector<int>& umax, int halfPatchSize) {
    int v, v0, vmax = floor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = ceil(halfPatchSize * std::sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = round(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

static void ICAngles(vector<Mat> imagePyramid, vector<KeyPoint>& keypoints, std::vector<int>& u_max, int half_k) {
    size_t ptidx, ptsize = keypoints.size();
    for (ptidx = 0; ptidx < ptsize; ptidx++) {
        int z = keypoints[ptidx].octave;
        int step = imagePyramid[z].size().width;
        const uchar* center = &imagePyramid[z].at<uchar>(cvRound(keypoints[ptidx].pt.y), cvRound(keypoints[ptidx].pt.x));

        int m_01 = 0, m_10 = 0;

        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        keypoints[ptidx].angle = fastAtan2((float)m_01, (float)m_10);

    }
}

static void computeKeypoints(vector<Mat>& imagePyramid, vector<KeyPoint>& keypoints, double scaleFactor,
    int nlevels, int border, int patchSize, int fastThreshold, 
    bool specifyNumberEachLevel,bool useFastScore, bool useHarrisScore,int harrisBlockSize,float harris_k ) {
    vector<int> nfeaturesPerLevel(nlevels);

    if (specifyNumberEachLevel) {
        cout << "Input number of feature points in " << nlevels << " levels." << endl;
        for (size_t level = 0; level < nlevels; level++) {
            cout << "Level " << level << ":";
            cin >> nfeaturesPerLevel[level];
        }

    }
    else {
        int nfeatures;
        cout << "Input total number of feature points!" << endl;
        cout << "Each level will get a default feature number based on scale." << endl;
        cout << "Total number:";
        cin >> nfeatures;
        float factor = (float)(1.0 / scaleFactor);

        float ndesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));
        int sumFeatures = 0;
        for (size_t level = 0; level < nlevels - 1; level++)
        {
            nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
            sumFeatures += nfeaturesPerLevel[level];
            ndesiredFeaturesPerScale *= factor;
        }
        nfeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
    }

    int halfPatchSize = patchSize / 2;
    vector<int> umax(halfPatchSize + 2);
    getUMax(umax, halfPatchSize);

    keypoints.clear();
    vector<KeyPoint> coarseKeypoints;
    vector<int> counters(nlevels);
    for (size_t level = 0; level < nlevels; level++) {
        int featuresNum = nfeaturesPerLevel[level];
        Mat img = imagePyramid[level];

        //fast detection
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        fd->detect(img, coarseKeypoints, Mat());
        KeyPointsFilter::runByImageBorder(coarseKeypoints, img.size(), border);
        if ((useFastScore) && (useHarrisScore == false)) {
            KeyPointsFilter::retainBest(coarseKeypoints, featuresNum);
        }
        else if ((useHarrisScore) && (useFastScore == false)) {
            HarrisResponses(imagePyramid, coarseKeypoints, harrisBlockSize, harris_k);
            KeyPointsFilter::retainBest(coarseKeypoints, featuresNum);
        }
        else {
            KeyPointsFilter::retainBest(coarseKeypoints, 2 * featuresNum);
        }
        //KeyPointsFilter::retainBest(coarseKeypoints, useFastScore ? 2 * featuresNum : featuresNum);
        counters[level] = coarseKeypoints.size();
        for (size_t i = 0; i < coarseKeypoints.size(); i++)
            coarseKeypoints[i].octave = level;
        std::copy(coarseKeypoints.begin(), coarseKeypoints.end(), back_inserter(keypoints));

    }
    
    if (keypoints.size() == 0) {
        return;
    }

    if (useFastScore == useHarrisScore) {
        HarrisResponses(imagePyramid, keypoints, harrisBlockSize, harris_k);
        vector<KeyPoint> newKeypoints;

        int offset = 0;
        int nkeypoints;
        
        for (size_t level = 0; level < nlevels; level++) {
            int featuresNum = nfeaturesPerLevel[level];
            nkeypoints = counters[level];
            coarseKeypoints.resize(nkeypoints);
            copy(keypoints.begin() + offset, keypoints.begin() + offset + nkeypoints, coarseKeypoints.begin());
            
            offset += nkeypoints;
            
            KeyPointsFilter::retainBest(coarseKeypoints, featuresNum);

            copy(coarseKeypoints.begin(), coarseKeypoints.end(), back_inserter(newKeypoints));
        }
        swap(keypoints, newKeypoints);
    }

    ICAngles(imagePyramid, keypoints, umax, halfPatchSize);
}

static void computeOrbDescriptors(vector<Mat> imagePyramid, vector<KeyPoint>& keypoints, Mat& descriptors,
    int nlevels, int gaussianSize, double gaussianSigma) {

    descriptors.create(keypoints.size(), 32, CV_8U);

    vector<Point> patternVec;
    const int npoints = 512;
    const Point* POINT = (const Point*)POINTPAIR;
    std::copy(POINT, POINT + npoints, back_inserter(patternVec));

    for (size_t level = 0; level < nlevels; level++)
    {
        Mat workingMat = imagePyramid[level];
        GaussianBlur(workingMat, workingMat, Size(gaussianSize, gaussianSize), gaussianSigma, gaussianSigma, BORDER_REFLECT_101);
    }

    size_t ptidx, ptsize = keypoints.size();
    for (ptidx = 0; ptidx < ptsize; ptidx++) {
        const KeyPoint& kpt = keypoints[ptidx];

        //compute angle
        float angle = kpt.angle;
        angle *= (float)(CV_PI / 180.f);
        float a = (float)cos(angle), b = (float)sin(angle);

        //get position information
        int octave = kpt.octave;
        int step = imagePyramid[octave].size().width;
        const uchar* center = &imagePyramid[octave].at<uchar>(cvRound(keypoints[ptidx].pt.y), cvRound(keypoints[ptidx].pt.x));
        float x, y;
        int ix, iy;
        const Point* pattern = &patternVec[0];

#define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )

        uchar* desc = descriptors.ptr<uchar>(ptidx);
        for (size_t i = 0; i < 32; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            desc[i] = (uchar)val;
        }
    }
}

void MY_ORB::detectAndCompute(Mat image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    if (image.type() != CV_8UC1)
        cvtColor(image, image, COLOR_BGR2GRAY);
    vector<Mat>imagePyramid;
    vector<float> layerScale(nlevels);
    makeImagePyramid(image, imagePyramid, layerScale, scaleFactor, nlevels, firstLevel);
    computeKeypoints(imagePyramid, keypoints, scaleFactor, nlevels, border, patchSize, fastThreshold, specifyNumberEachLevel,useFastScore,useHarrisScore,harrisBlockSize,harris_k);
    computeOrbDescriptors(imagePyramid, keypoints, descriptors, nlevels, gaussianSize, gaussianSigma);
    
    for (size_t ptidx = 0; ptidx < keypoints.size(); ptidx++) {
        float scale = layerScale[keypoints[ptidx].octave];
        keypoints[ptidx].pt *= scale;
    }
    
}

int main() {
    Mat image = imread("d:\\1.jpg", IMREAD_COLOR);
    if (image.empty())return -1;
    resize(image, image, Size(4578, 3052), INTER_LANCZOS4);
    //MY_ORB orb(
    //    2,        //scaleFactor : 缩小因子，图像每层较上一层缩小的比例
    //    2,          //nlevels : 金字塔总层数
    //    1,          //firstLevel : 原图放在金字塔第几层。如果不放在第0层，则[0,firstLevel)层的图像由原图生采样得到
    //    31,         //border : 距离边界太近的粗角点将被删除
    //    31,         //patchSize : 特征点角度计算圆形区域的直径(包括特征点本身)
    //    20,         //fastThreshold : fast角点检测阈值
    //    false,      //specifyNumberEachLevel : 是否指定金字塔每层图像应检测出的特征点数目。如果为false，则接收总特征点数，按照默认方式为各层分配特征点容量。
    //    true,       //useFastScore : 是否使用Fast score进行粗特征点筛选。
    //    false,      //useHarrisScore : 是否使用Harris score进行粗特征点筛选。
    //                //如果useFastScore和useHarrisScore同时为false或者同时为true，则同时采用两种分数
    //    7,          //harrisBlockSize : harris分数ROI区域的长度。
    //    0.04f,       //harris_k : 计算harris score的k值，paper建议一般为0.04-0.06。
    //    7,           //计算描述子前，高斯模糊kernel size
    //    2.0         //计算描述子前，高斯模糊kernel sigma
    //);

    //vector<KeyPoint>keypoints;
    //Mat descriptors;
    //orb.detectAndCompute(image, keypoints, descriptors);
    //for (int i = 0; i < keypoints.size(); i++) {
    //    cout << keypoints[i].pt << " " << keypoints[i].response << " "<<keypoints[i].octave<<endl;
    //}

    Ptr<ORB> orb = ORB::create(10, 2, 1, 31, 0, 2, ORB::FAST_SCORE, 31, 20);
    vector<KeyPoint>keypoints;
    Mat descriptors;
    orb->detectAndCompute(image, Mat(), keypoints, descriptors);
    for (int i = 0; i < keypoints.size(); i++) {
        cout << keypoints[i].pt << " " << keypoints[i].response << " "<<keypoints[i].octave<<endl;
    }
    
}
