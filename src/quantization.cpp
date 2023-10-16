// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>


#ifdef MESHOPT_DOUBLE_PRECISION

unsigned short meshopt_quantizeHalf(meshopt_float v)
{
    union { meshopt_float d; unsigned long long ull; } u = {v};
    unsigned long long ull = u.ull;

    int s = (ull >> 48) & 0x8000;
    unsigned long long em = ull & 0x7fffffffffffffffULL;

    // bias exponent and round to nearest; 1008 is relative exponent bias (1023-15)
    int h = (em - (1008ULL << 52) + (1ULL << 41)) >> 42;

    // underflow: flush to zero; 1009 encodes exponent -14
    h = (em < (1009ULL << 52)) ? 0 : h;

    // overflow: infinity; 1039 encodes exponent 16
    h = (em >= (1039ULL << 52)) ? 0x7c00 : h;

    // NaN; note that we convert all types of NaN to qNaN
    h = (em > (2047ULL << 52)) ? 0x7e00 : h;

    return (unsigned short)(s | h);
}

meshopt_float meshopt_quantizeFloat(meshopt_float v, int N)
{
    assert(N >= 0 && N <= 52);

    union { meshopt_float d; unsigned long long ull; } u = {v};
    unsigned long long ull = u.ull;

    const unsigned long long mask = (1ULL << (52 - N)) - 1;
    const unsigned long long round = (1ULL << (52 - N)) >> 1;

    unsigned long long e = ull & 0x7FF0000000000000ULL;
    unsigned long long rull = (ull + round) & ~mask;

    // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
    ull = e == 0x7FF0000000000000ULL ? ull : rull;

    // flush denormals to zero
    ull = e == 0 ? 0 : ull;

    u.ull = ull;
    return u.d;
}

meshopt_float meshopt_dequantizeHalf(unsigned short h)
{
    unsigned long long s = (unsigned long long)(h & 0x8000) << 48;
    int em = h & 0x7fff;

    // bias exponent and pad mantissa with 0; 1008 is relative exponent bias (1023-15)
    unsigned long long r = (unsigned long long)(em + (1008 << 10)) << 42;

    // denormal: flush to zero
    r = (em < (1 << 10)) ? 0 : r;

    // infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
    // 1008 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 2047
    r += (em >= (31 << 10)) ? (1008ULL << 52) : 0;

    union { meshopt_float d; unsigned long long ull; } u;
    u.ull = s | r;
    return u.d;
}

#else

unsigned short meshopt_quantizeHalf(meshopt_float v)
{
	union { meshopt_float f; unsigned int ui; } u = {v};
	unsigned int ui = u.ui;

	int s = (ui >> 16) & 0x8000;
	int em = ui & 0x7fffffff;

	// bias exponent and round to nearest; 112 is relative exponent bias (127-15)
	int h = (em - (112 << 23) + (1 << 12)) >> 13;

	// underflow: flush to zero; 113 encodes exponent -14
	h = (em < (113 << 23)) ? 0 : h;

	// overflow: infinity; 143 encodes exponent 16
	h = (em >= (143 << 23)) ? 0x7c00 : h;

	// NaN; note that we convert all types of NaN to qNaN
	h = (em > (255 << 23)) ? 0x7e00 : h;

	return (unsigned short)(s | h);
}

meshopt_float meshopt_quantizeFloat(meshopt_float v, int N)
{
	assert(N >= 0 && N <= 23);

	union { meshopt_float f; unsigned int ui; } u = {v};
	unsigned int ui = u.ui;

	const int mask = (1 << (23 - N)) - 1;
	const int round = (1 << (23 - N)) >> 1;

	int e = ui & 0x7f800000;
	unsigned int rui = (ui + round) & ~mask;

	// round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
	ui = e == 0x7f800000 ? ui : rui;

	// flush denormals to zero
	ui = e == 0 ? 0 : ui;

	u.ui = ui;
	return u.f;
}

meshopt_float meshopt_dequantizeHalf(unsigned short h)
{
	unsigned int s = unsigned(h & 0x8000) << 16;
	int em = h & 0x7fff;

	// bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
	int r = (em + (112 << 10)) << 13;

	// denormal: flush to zero
	r = (em < (1 << 10)) ? 0 : r;

	// infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
	// 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
	r += (em >= (31 << 10)) ? (112 << 23) : 0;

	union { meshopt_float f; unsigned int ui; } u;
	u.ui = s | r;
	return u.f;
}

#endif
