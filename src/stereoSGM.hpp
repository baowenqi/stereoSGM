#pragma once
#include <stdint.h>

class stereoSGM
{
    public:
    int8_t  *m_dispMap;
    typedef enum e_status {ERROR, SUCCESS} status_t;

             stereoSGM(int32_t i_imgWidth, int32_t i_imgHeight, int32_t i_imgDisp, int8_t i_P1, int8_t i_P2);
            ~stereoSGM();
    e_status compute(uint8_t *imgLeft, uint8_t *imgRight);

    private:
    int32_t  m_imgWidth;
    int32_t  m_imgHeight;
    int32_t  m_imgDisp;
    int8_t   m_P1;
    int8_t   m_P2;
    int8_t   m_invalid;
    int8_t   m_threshold;
    typedef enum e_direction {L0, L1, L2, L3, L4, L5, L6, L7} direction_t;
    typedef enum e_pickLR {LEFT, RIGHT} pickLR_t;

    status_t f_censusTransform5x5(uint8_t *src, int32_t *dst);
    status_t f_getMatchCost(int32_t *ctLeft, int32_t *ctRight, int32_t *matchCost);
    status_t f_getPathCost(int8_t *matchCost, int8_t *pathCost, e_direction dir);
    status_t f_aggregateCost(int8_t *matchCost, int32_t *sumCost);
    status_t f_pickDisparity(int32_t *sumCost, int8_t *dispMap, pickLR_t lr);
    status_t f_checkLeftRight(int8_t *dispLeft, int8_t *dispRight, int8_t *dispMap);
    status_t f_medianFilter3x3(int8_t *src, int8_t *dst);
};

