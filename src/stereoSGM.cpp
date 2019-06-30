#include <assert.h>
#include <stdint.h>
#include "stereoSGM.hpp"

using namespace std;

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
