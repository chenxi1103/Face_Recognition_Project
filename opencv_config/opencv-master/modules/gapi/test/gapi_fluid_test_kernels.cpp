// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"

#include <iomanip>
#include "gapi_fluid_test_kernels.hpp"
#include <opencv2/gapi/core.hpp>

namespace cv
{
namespace gapi_test_kernels
{

GAPI_FLUID_KERNEL(FAddSimple, TAddSimple, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View   &a,
                    const cv::gapi::fluid::View   &b,
                          cv::gapi::fluid::Buffer &o)
    {
        // std::cout << "AddSimple {{{\n";
        // std::cout << "  a - "; a.debug(std::cout);
        // std::cout << "  b - "; b.debug(std::cout);
        // std::cout << "  o - "; o.debug(std::cout);

        const uint8_t* in1 = a.InLine<uint8_t>(0);
        const uint8_t* in2 = b.InLine<uint8_t>(0);
              uint8_t* out = o.OutLine<uint8_t>();

        // std::cout << "a: ";
        // for (int i = 0, w = a.length(); i < w; i++)
        // {
        //     std::cout << std::setw(4) << int(in1[i]);
        // }
        // std::cout << "\n";

        // std::cout << "b: ";
        // for (int i = 0, w = a.length(); i < w; i++)
        // {
        //     std::cout << std::setw(4) << int(in2[i]);
        // }
        // std::cout << "\n";

        for (int i = 0, w = a.length(); i < w; i++)
        {
            out[i] = in1[i] + in2[i];
        }

        // std::cout << "}}} " << std::endl;;
    }
};

GAPI_FLUID_KERNEL(FAddCSimple, TAddCSimple, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &in,
                    const int                      cval,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            //std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                //std::cout << std::setw(4) << int(in_row[i]);
                out_row[i] = static_cast<uint8_t>(in_row[i] + cval);
            }
            //std::cout << std::endl;
        }
    }
};

GAPI_FLUID_KERNEL(FAddScalar, TAddScalar, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &in,
                    const cv::Scalar              &cval,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row[i]);
                out_row[i] = static_cast<uint8_t>(in_row[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

GAPI_FLUID_KERNEL(FAddScalarToMat, TAddScalarToMat, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::Scalar              &cval,
                    const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row  = in .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = in.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row[i]);
                out_row[i] = static_cast<uint8_t>(in_row[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

template<int kernelSize, int lpi = 1>
static void runBlur(const cv::gapi::fluid::View& src, cv::gapi::fluid::Buffer& dst)
{
    const auto borderSize = (kernelSize - 1) / 2;
    const unsigned char* ins[kernelSize];

    for (int l = 0; l < lpi; l++)
    {
        for (int i = 0; i < kernelSize; i++)
        {
            ins[i] = src.InLine<unsigned char>(i - borderSize + l);
        }

        auto out = dst.OutLine<unsigned char>(l);
        const auto width = dst.length();

        for (int w = 0; w < width; w++)
        {
            float res = 0.0f;
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = -borderSize; j < borderSize + 1; j++)
                {
                    res += ins[i][w+j];
                }
            }
            out[w] = static_cast<unsigned char>(std::rint(res / (kernelSize * kernelSize)));
        }
    }
}

GAPI_FLUID_KERNEL(FBlur1x1, TBlur1x1, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }
};

GAPI_FLUID_KERNEL(FBlur3x3, TBlur3x3, false)
{
    static const int Window = 3;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, to_own(borderValue)};
    }
};

GAPI_FLUID_KERNEL(FBlur5x5, TBlur5x5, false)
{
    static const int Window = 5;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, to_own(borderValue)};
    }
};

GAPI_FLUID_KERNEL(FBlur3x3_2lpi, TBlur3x3_2lpi, false)
{
    static const int Window = 3;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window, LPI>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, to_own(borderValue)};
    }
};

GAPI_FLUID_KERNEL(FBlur5x5_2lpi, TBlur5x5_2lpi, false)
{
    static const int Window = 5;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View &src, int /*borderType*/,
                    cv::Scalar /*borderValue*/, cv::gapi::fluid::Buffer &dst)
    {
        runBlur<Window, LPI>(src, dst);
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc &/*src*/, int borderType, cv::Scalar borderValue)
    {
        return { borderType, to_own(borderValue )};
    }
};

GAPI_FLUID_KERNEL(FIdentity, TId, false)
{
    static const int Window = 3;

    static void run(const cv::gapi::fluid::View   &a,
                          cv::gapi::fluid::Buffer &o)
    {
        const uint8_t* in[3] = {
            a.InLine<uint8_t>(-1),
            a.InLine<uint8_t>( 0),
            a.InLine<uint8_t>(+1)
        };
        uint8_t* out = o.OutLine<uint8_t>();

        // ReadFunction3x3(in, a.length());
        for (int i = 0, w = a.length(); i < w; i++)
        {
            out[i] = in[1][i];
        }
    }

    static gapi::fluid::Border getBorder(const cv::GMatDesc &)
    {
        return { cv::BORDER_REPLICATE, cv::gapi::own::Scalar{} };
    }
};

GAPI_FLUID_KERNEL(FId7x7, TId7x7, false)
{
    static const int Window = 7;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &a,
                          cv::gapi::fluid::Buffer &o)
    {
        for (int l = 0, lpi = o.lpi(); l < lpi; l++)
        {
            const uint8_t* in[Window] = {
                a.InLine<uint8_t>(-3 + l),
                a.InLine<uint8_t>(-2 + l),
                a.InLine<uint8_t>(-1 + l),
                a.InLine<uint8_t>( 0 + l),
                a.InLine<uint8_t>(+1 + l),
                a.InLine<uint8_t>(+2 + l),
                a.InLine<uint8_t>(+3 + l),
            };
            uint8_t* out = o.OutLine<uint8_t>(l);

            // std::cout << "Id7x7 " << l << " of " << lpi << " {{{\n";
            // std::cout << "  a - "; a.debug(std::cout);
            // std::cout << "  o - "; o.debug(std::cout);
            // std::cout << "}}} " << std::endl;;

            // // std::cout << "Id7x7 at " << a.y() << "/L" << l <<  " {{{" << std::endl;
            // for (int j = 0; j < Window; j++)
            // {
            //     // std::cout << std::setw(2) << j-(Window-1)/2 << ": ";
            //     for (int i = 0, w = a.length(); i < w; i++)
            //         std::cout << std::setw(4) << int(in[j][i]);
            //     std::cout << std::endl;
            // }
            // std::cout << "}}}" << std::endl;

            for (int i = 0, w = a.length(); i < w; i++)
                out[i] = in[(Window-1)/2][i];
        }
    }

    static cv::gapi::fluid::Border getBorder(const cv::GMatDesc&/* src*/)
    {
        return { cv::BORDER_REPLICATE, cv::gapi::own::Scalar{} };
    }
};

GAPI_FLUID_KERNEL(FPlusRow0, TPlusRow0, true)
{
    static const int Window = 1;

    static void initScratch(const cv::GMatDesc            &in,
                                  cv::gapi::fluid::Buffer &scratch)
    {
        cv::Size scratch_size{in.size.width, 1};
        cv::gapi::fluid::Buffer buffer(in.withSize(scratch_size));
        scratch = std::move(buffer);
    }

    static void resetScratch(cv::gapi::fluid::Buffer &scratch)
    {
        // FIXME: only 1 line can be used!
        uint8_t* out_row = scratch.OutLine<uint8_t>();
        for (int i = 0, w = scratch.length(); i < w; i++)
        {
            out_row[i] = 0;
        }
    }

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &out,
                          cv::gapi::fluid::Buffer &scratch)
    {
        const uint8_t* in_row  = in     .InLine <uint8_t>(0);
              uint8_t* out_row = out    .OutLine<uint8_t>();
              uint8_t* tmp_row = scratch.OutLine<uint8_t>();

        if (in.y() == 0)
        {
            // Copy 1st row to scratch buffer
            for (int i = 0, w = in.length(); i < w; i++)
            {
                out_row[i] = in_row[i];
                tmp_row[i] = in_row[i];
            }
        }
        else
        {
            // Output is 1st row + in
            for (int i = 0, w = in.length(); i < w; i++)
            {
                out_row[i] = in_row[i] + tmp_row[i];
            }
        }
    }
};

GAPI_FLUID_KERNEL(FTestSplit3, cv::gapi::core::GSplit3, false)
{
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View   &in,
                          cv::gapi::fluid::Buffer &o1,
                          cv::gapi::fluid::Buffer &o2,
                          cv::gapi::fluid::Buffer &o3)
    {
        // std::cout << "Split3  {{{\n";
        // std::cout << "  a - "; in.debug(std::cout);
        // std::cout << "  1 - "; o1.debug(std::cout);
        // std::cout << "  2 - "; o2.debug(std::cout);
        // std::cout << "  3 - "; o3.debug(std::cout);
        // std::cout << "}}} " << std::endl;;

        const uint8_t* in_rgb = in.InLine<uint8_t>(0);
              uint8_t* out_r  = o1.OutLine<uint8_t>();
              uint8_t* out_g  = o2.OutLine<uint8_t>();
              uint8_t* out_b  = o3.OutLine<uint8_t>();

        for (int i = 0, w = in.length(); i < w; i++)
        {
            out_r[i] = in_rgb[3*i];
            out_g[i] = in_rgb[3*i+1];
            out_b[i] = in_rgb[3*i+2];
        }
    }
};

GAPI_FLUID_KERNEL(FSum2MatsAndScalar, TSum2MatsAndScalar, false)
{
    static const int Window = 1;
    static const int LPI    = 2;

    static void run(const cv::gapi::fluid::View   &a,
                    const cv::Scalar              &cval,
                    const cv::gapi::fluid::View   &b,
                          cv::gapi::fluid::Buffer &out)
    {
        for (int l = 0, lpi = out.lpi(); l < lpi; l++)
        {
            const uint8_t* in_row1  = a .InLine <uint8_t>(l);
            const uint8_t* in_row2  = b .InLine <uint8_t>(l);
                  uint8_t* out_row = out.OutLine<uint8_t>(l);
            std::cout << "l=" << l << ": ";
            for (int i = 0, w = a.length(); i < w; i++)
            {
                std::cout << std::setw(4) << int(in_row1[i]);
                std::cout << std::setw(4) << int(in_row2[i]);
                out_row[i] = static_cast<uint8_t>(in_row1[i] + in_row2[i] + cval[0]);
            }
            std::cout << std::endl;
        }
    }
};

cv::gapi::GKernelPackage fluidTestPackage = cv::gapi::kernels
        <FAddSimple
        ,FAddCSimple
        ,FAddScalar
        ,FAddScalarToMat
        ,FBlur1x1
        ,FBlur3x3
        ,FBlur5x5
        ,FBlur3x3_2lpi
        ,FBlur5x5_2lpi
        ,FIdentity
        ,FId7x7
        ,FPlusRow0
        ,FSum2MatsAndScalar
        ,FTestSplit3
        >();
} // namespace gapi_test_kernels
} // namespace cv
