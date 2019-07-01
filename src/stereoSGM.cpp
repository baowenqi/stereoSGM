#include <assert.h>
#include <stdint.h>
#include "stereoSGM.hpp"

using namespace std;

// ------------------------------------------------- //
// census transform over a 5x5 window.               //
// it's caller's job to pad the border by 2 elements //
// a 5x5 window has 24 elements comparing with the   //
// center element so the output can be encoded in    //
// int32_t (lower 24 bits are useful).               //
// the mapping of the window elements position and   //
// output bit position (center-skipped) is like:     //
//      0    1    2    3    4                        //
//   |----+----+----+----+----|                      //
// 0 | 23 | 22 | 21 | 20 | 19 |                      // 
//   |----+----+----+----+----|                      //
// 1 | 18 | 17 | 16 | 15 | 14 |                      // 
//   |----+----+----+----+----|                      //
// 2 | 13 | 12 |  * | 11 | 10 |                      // 
//   |----+----+----+----+----|                      //
// 3 |  9 |  8 |  7 |  6 |  5 |                      // 
//   |----+----+----+----+----|                      //
// 4 |  4 |  3 |  2 |  1 |  0 |                      // 
//   |----+----+----+----+----|                      //
// if window(i, j) > window(2, 2), then its corres-  //
// ponding output bit is 1, otherwise it's 0.        //
// ------------------------------------------------- //
stereoSGM::status_t stereoSGM::f_censusTransform5x5
(
    uint8_t *src,
    int32_t *dst
)
{
    // --------------------------------------------- //
    // as the census transform window is 5x5, need   //
    // to pad 2 to the border so the src line pitch  //
    // is 4-element larger then dst                  // 
    // --------------------------------------------- //
    const int32_t dstLinePitch = m_imgWidth;
    const int32_t srcLinePitch = dstLinePitch + ctWinRad * 2;

    for(int dy = 0; dy < m_imgHeight; dy++)
    {
        for(int dx = 0; dx < m_imgWidth; dx++)
        {
            int32_t dstData = 0;
            
            // ------------------------------------- //
            // dx and dy map to top-left corner of   //
            // the window so that + winRad to get    //
            // center                                //
            // ------------------------------------- //
            int32_t srcCenterOfst = (dy + ctWinRad) * srcLinePitch + (dx + ctWinRad);
            uint8_t srcCenter = *(src + srcCenterOfst);

            for(int wy = 0; wy < ctWinSize; wy++)
            {
                for(int wx = 0; wx < ctWinSize; wx++)
                {
                    int32_t bitPos = ctDataMsb - (wy * ctWinSize + wx);

                    // ----------------------------- //
                    // skip the center element       //
                    // ----------------------------- //
                    if(bitPos != 11)
                    {
                        int32_t srcDataOfst = (dy + wy) * srcLinePitch + (dx + wx);
                        uint8_t srcData = *(src + srcDataOfst);
                        // -------------------------------------------- //
                        // for the bit position less then 11 (center),  //
                        // +1 to adjust the position, fit the hole of   //
                        // center                                       //
                        // -------------------------------------------- //
                        bitPos = bitPos < 11 ? bitPos + 1 : bitPos;
                        dstData |= (srcData > srcCenter) << bitPos;
                    }
                }
            }
        }
    }

    return SUCCESS;
}

// ------------------------------------------------- //
// hamming distance is the number of different bits  //
// between src1 and src2. so we xor src1 and src2    //
// first and count the number of 1s. As the prestage //
// is 5x5 census transform, only 24 useful bits, the //
// max distance is 24 which can be store in 1 byte.  //
// ------------------------------------------------- //
int8_t stereoSGM::f_getHammingDistance
(
    int32_t src1,
    int32_t src2
)
{
    int32_t diff = src1 ^ src2;

    int8_t distance = 0;
    for(int i = 0; i < ctDataLen; i++) distance += (diff >> i) & 1;

    return distance;
}

// ------------------------------------------------- //
// the pixel-wise matching cost is the hamming dist- //
// ance between L(x) and R(x+d), here d ranges from  //
// 0 to max disparity and forms a matching cost cube //
// matchCost(h * w * d), d is the inner-most dim.    //
// ------------------------------------------------- //
stereoSGM::status_t stereoSGM::f_getMatchCost
(
    int32_t *ctLeft,
    int32_t *ctRight,
    stereoSGMCostCube<int8_t> &matchCost
)
{
    const int32_t matchCostLP = m_imgWidth * m_imgDisp;

    for(int y = 0; y < m_imgHeight; y++)
    {
        for(int x = 0; x < m_imgWidth; x++)
        {
            int32_t leftData = *(ctLeft + y * m_imgWidth + x);
            for(int d = 0; d < m_imgDisp; d++)
            {
                int32_t rightData = 0;
                // ---------------------------------------- //
                // we need to make sure the right pixel is  //
                // not out-of-boundary, otherwise use zero  //
                // ---------------------------------------- //
                if(x - d >= 0) rightData = *(ctRight + y * m_imgWidth + x - d);
                matchCost.set(x, y, d, f_getHammingDistance(leftData, rightData));
            }
        }
    }
    return SUCCESS;
}

stereoSGM::stereoSGM
(
    int32_t i_imgWidth,
    int32_t i_imgHeight,
    int32_t i_imgDisp,
    int8_t  i_P1,
    int8_t  i_P2
) :
m_imgWidth(i_imgWidth),
m_imgHeight(i_imgHeight),
m_imgDisp(i_imgDisp),
m_P1(i_P1),
m_P2(i_P2)
{
    // --------------------------------------------- //
    // the disparity range should be multiple of 16  //
    // in this example, disparity is 64              //
    // make sure P1 is smaller than P2 (constant)    //
    // --------------------------------------------- //
    assert(m_imgDisp % 16 == 0);
    assert(m_P1 < m_P2);

    m_dispMap = new int8_t[m_imgWidth * m_imgHeight];
}

stereoSGM::~stereoSGM()
{
    if(m_dispMap != nullptr) delete []m_dispMap;
}
