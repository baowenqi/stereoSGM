#pragma once

class stereSGM
{
    private:
    int32_t  m_imgWidth;
    int32_t  m_imgHeight;
    int32_t  m_imgDisp;
    int8_t   m_P1;
    int8_t   m_P2;
    int8_t   m_invalid;
    int8_t   m_threshold;
    enum     e_direction {L0, L1, L2, L3, L4, L5, L6, L7};
    enum     e_pickLR {LEFT, RIGHT};

    e_status f_censusTransform5x5(uint8_t *src, int32_t *dst);
    e_status f_getMatchCost(int32_t *ctLeft, int32_t *ctRight, int32_t *matchCost);
    e_status f_getPathCost(int8_t *matchCost, int8_t *pathCost, e_direction dir);
    e_status f_aggregateCost(int8_t *matchCost, int32_t *sumCost);
    e_status f_pickDisparity(int32_t *sumCost, int8_t *dispMap, e_pickLR lr);
    e_status f_checkLeftRight(int8_t *dispLeft, int8_t *dispRight, int8_t *dispMap);
    e_status f_medianFilter3x3(int8_t *src, int8_t *dst);

    public:
    int8_t  *m_dispMap;
    enum     e_status {ERROR, SUCCESS};

             stereoSGM(int32_t i_imgWidth, int32_t i_imgHeight, int32_t i_imgDisp, int8_t i_P1, int8_t i_P2);
            ~stereoSGM();
    e_status compute(uint8_t *imgLeft, uint8_t *imgRight);
};

