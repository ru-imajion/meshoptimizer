// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <float.h>
#include <math.h>
#include <string.h>

static cgltf_float getDelta(const Attr& l, const Attr& r, cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return std::max(std::max(gp_fabs(l.f[0] - r.f[0]), gp_fabs(l.f[1] - r.f[1])), gp_fabs(l.f[2] - r.f[2]));

	case cgltf_animation_path_type_rotation:
		return gp_acos(std::min((cgltf_float)1.f, gp_fabs(l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3])));

	case cgltf_animation_path_type_scale:
		return std::max(std::max(gp_fabs(l.f[0] / r.f[0] - 1), gp_fabs(l.f[1] / r.f[1] - 1)), gp_fabs(l.f[2] / r.f[2] - 1));

	case cgltf_animation_path_type_weights:
		return gp_fabs(l.f[0] - r.f[0]);

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static cgltf_float getDeltaTolerance(cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return 0.0001f; // 0.1mm linear

	case cgltf_animation_path_type_rotation:
		return 0.1f * (3.1415926f / 180.f); // 0.1 degrees

	case cgltf_animation_path_type_scale:
		return 0.001f; // 0.1% ratio

	case cgltf_animation_path_type_weights:
		return 0.001f; // 0.1% linear

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static Attr interpolateLinear(const Attr& l, const Attr& r, cgltf_float t, cgltf_animation_path_type type)
{
	if (type == cgltf_animation_path_type_rotation)
	{
		// Approximating slerp, https://zeux.io/2015/07/23/approximating-slerp/
		// We also handle quaternion double-cover
		cgltf_float ca = l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3];

		cgltf_float d = gp_fabs(ca);
		cgltf_float A = 1.0904f + d * (-3.2452f + d * (3.55645f - d * 1.43519f));
		cgltf_float B = 0.848013f + d * (-1.06021f + d * 0.215638f);
		cgltf_float k = A * (t - 0.5f) * (t - 0.5f) + B;
		cgltf_float ot = t + t * (t - 0.5f) * (t - 1) * k;

		cgltf_float t0 = 1 - ot;
		cgltf_float t1 = ca > 0 ? ot : -ot;

		Attr lerp = {{
		    l.f[0] * t0 + r.f[0] * t1,
		    l.f[1] * t0 + r.f[1] * t1,
		    l.f[2] * t0 + r.f[2] * t1,
		    l.f[3] * t0 + r.f[3] * t1,
		}};

		cgltf_float len = gp_sqrt(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.f)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}

		return lerp;
	}
	else
	{
		Attr lerp = {{
		    l.f[0] * (1 - t) + r.f[0] * t,
		    l.f[1] * (1 - t) + r.f[1] * t,
		    l.f[2] * (1 - t) + r.f[2] * t,
		    l.f[3] * (1 - t) + r.f[3] * t,
		}};

		return lerp;
	}
}

static Attr interpolateHermite(const Attr& v0, const Attr& t0, const Attr& v1, const Attr& t1, cgltf_float t, cgltf_float dt, cgltf_animation_path_type type)
{
	cgltf_float s0 = 1 + t * t * (2 * t - 3);
	cgltf_float s1 = t + t * t * (t - 2);
	cgltf_float s2 = 1 - s0;
	cgltf_float s3 = t * t * (t - 1);

	cgltf_float ts1 = dt * s1;
	cgltf_float ts3 = dt * s3;

	Attr lerp = {{
	    s0 * v0.f[0] + ts1 * t0.f[0] + s2 * v1.f[0] + ts3 * t1.f[0],
	    s0 * v0.f[1] + ts1 * t0.f[1] + s2 * v1.f[1] + ts3 * t1.f[1],
	    s0 * v0.f[2] + ts1 * t0.f[2] + s2 * v1.f[2] + ts3 * t1.f[2],
	    s0 * v0.f[3] + ts1 * t0.f[3] + s2 * v1.f[3] + ts3 * t1.f[3],
	}};

	if (type == cgltf_animation_path_type_rotation)
	{
		cgltf_float len = gp_sqrt(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.f)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}
	}

	return lerp;
}

static void resampleKeyframes(std::vector<Attr>& data, const std::vector<cgltf_float>& input, const std::vector<Attr>& output, cgltf_animation_path_type type, cgltf_interpolation_type interpolation, size_t components, int frames, cgltf_float mint, int freq)
{
	size_t cursor = 0;

	for (int i = 0; i < frames; ++i)
	{
		cgltf_float time = mint + cgltf_float(i) / freq;

		while (cursor + 1 < input.size())
		{
			cgltf_float next_time = input[cursor + 1];

			if (next_time > time)
				break;

			cursor++;
		}

		if (cursor + 1 < input.size())
		{
			cgltf_float cursor_time = input[cursor + 0];
			cgltf_float next_time = input[cursor + 1];

			cgltf_float range = next_time - cursor_time;
			cgltf_float inv_range = (range == 0.f) ? 0.f : 1.f / (next_time - cursor_time);
			cgltf_float t = std::max((cgltf_float)0.f, std::min((cgltf_float)1.f, (time - cursor_time) * inv_range));

			for (size_t j = 0; j < components; ++j)
			{
				switch (interpolation)
				{
				case cgltf_interpolation_type_linear:
				{
					const Attr& v0 = output[(cursor + 0) * components + j];
					const Attr& v1 = output[(cursor + 1) * components + j];
					data.push_back(interpolateLinear(v0, v1, t, type));
				}
				break;

				case cgltf_interpolation_type_step:
				{
					const Attr& v = output[cursor * components + j];
					data.push_back(v);
				}
				break;

				case cgltf_interpolation_type_cubic_spline:
				{
					const Attr& v0 = output[(cursor * 3 + 1) * components + j];
					const Attr& b0 = output[(cursor * 3 + 2) * components + j];
					const Attr& a1 = output[(cursor * 3 + 3) * components + j];
					const Attr& v1 = output[(cursor * 3 + 4) * components + j];
					data.push_back(interpolateHermite(v0, b0, v1, a1, t, range, type));
				}
				break;

				default:
					assert(!"Unknown interpolation type");
				}
			}
		}
		else
		{
			size_t offset = (interpolation == cgltf_interpolation_type_cubic_spline) ? cursor * 3 + 1 : cursor;

			for (size_t j = 0; j < components; ++j)
			{
				const Attr& v = output[offset * components + j];
				data.push_back(v);
			}
		}
	}
}

static cgltf_float getMaxDelta(const std::vector<Attr>& data, cgltf_animation_path_type type, int frames, const Attr* value, size_t components)
{
	assert(data.size() == frames * components);

	cgltf_float result = 0;

	for (int i = 0; i < frames; ++i)
	{
		for (size_t j = 0; j < components; ++j)
		{
			cgltf_float delta = getDelta(value[j], data[i * components + j], type);

			result = (result < delta) ? delta : result;
		}
	}

	return result;
}

static void getBaseTransform(Attr* result, size_t components, cgltf_animation_path_type type, cgltf_node* node)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		memcpy(result->f, node->translation, 3 * sizeof(cgltf_float));
		break;

	case cgltf_animation_path_type_rotation:
		memcpy(result->f, node->rotation, 4 * sizeof(cgltf_float));
		break;

	case cgltf_animation_path_type_scale:
		memcpy(result->f, node->scale, 3 * sizeof(cgltf_float));
		break;

	case cgltf_animation_path_type_weights:
		if (node->weights_count)
		{
			assert(node->weights_count == components);
			memcpy(result->f, node->weights, components * sizeof(cgltf_float));
		}
		else if (node->mesh && node->mesh->weights_count)
		{
			assert(node->mesh->weights_count == components);
			memcpy(result->f, node->mesh->weights, components * sizeof(cgltf_float));
		}
		break;

	default:
		assert(!"Unknown animation path");
	}
}

static cgltf_float getWorldScale(cgltf_node* node)
{
	cgltf_float transform[16];
	cgltf_node_transform_world(node, transform);

	// 3x3 determinant computes scale^3
	cgltf_float a0 = transform[5] * transform[10] - transform[6] * transform[9];
	cgltf_float a1 = transform[4] * transform[10] - transform[6] * transform[8];
	cgltf_float a2 = transform[4] * transform[9] - transform[5] * transform[8];
	cgltf_float det = transform[0] * a0 - transform[1] * a1 + transform[2] * a2;

	return gp_pow(gp_fabs(det), 1.f / 3.f);
}

void processAnimation(Animation& animation, const Settings& settings)
{
	cgltf_float mint = GP_FLT_MAX, maxt = 0;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		const Track& track = animation.tracks[i];
		assert(!track.time.empty());

		mint = std::min(mint, track.time.front());
		maxt = std::max(maxt, track.time.back());
	}

	mint = std::min(mint, maxt);

	// round the number of frames to nearest but favor the "up" direction
	// this means that at 10 Hz resampling, we will try to preserve the last frame <10ms
	// but if the last frame is <2ms we favor just removing this data
	int frames = 1 + int((maxt - mint) * settings.anim_freq + 0.8f);

	animation.start = mint;
	animation.frames = frames;

	std::vector<Attr> base;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		Track& track = animation.tracks[i];

		std::vector<Attr> result;
		resampleKeyframes(result, track.time, track.data, track.path, track.interpolation, track.components, frames, mint, settings.anim_freq);

		track.time.clear();
		track.data.swap(result);

		cgltf_float tolerance = getDeltaTolerance(track.path);

		// translation tracks use world space tolerance; in the future, we should compute all errors as linear using hierarchy
		if (track.node && track.node->parent && track.path == cgltf_animation_path_type_translation)
		{
			cgltf_float scale = getWorldScale(track.node->parent);
			tolerance /= scale == 0.f ? 1.f : scale;
		}

		cgltf_float deviation = getMaxDelta(track.data, track.path, frames, &track.data[0], track.components);

		if (deviation <= tolerance)
		{
			// track is constant (equal to first keyframe), we only need the first keyframe
			track.constant = true;
			track.data.resize(track.components);

			// track.dummy is true iff track redundantly sets up the value to be equal to default node transform
			base.resize(track.components);
			getBaseTransform(&base[0], track.components, track.path, track.node);

			track.dummy = getMaxDelta(track.data, track.path, 1, &base[0], track.components) <= tolerance;
		}
	}
}
