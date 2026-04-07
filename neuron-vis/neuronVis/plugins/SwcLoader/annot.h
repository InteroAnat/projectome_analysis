#ifndef _FNT_ANNOT_H_
#define _FNT_ANNOT_H_

#include <cstdint>
#include <string>

struct AnnotItem {
	uint32_t par_idx;
	uint64_t id;

	int color_r;
	int color_g;
	int color_b;

	std::string abbrev;
	std::string name;

	/* only for internal usage */
	uint32_t child_idx;
	uint32_t child_num;
	int row;
};

#endif

