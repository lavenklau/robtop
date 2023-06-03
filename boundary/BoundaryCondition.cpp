#include "BoundaryCondition.h"

std::ostream& static_range::operator<<(std::ostream& os, static_range::rangeUnion& ru) {
	for (int i = 0; i < ru.closed_spheres.size(); i++) { os << ru.closed_spheres[i].textInfo(); }
	for (int i = 0; i < ru.spheres.size(); i++) { os << ru.spheres[i].textInfo(); }
	for (int i = 0; i < ru.closed_half_spaces.size(); i++) { os << ru.closed_half_spaces[i].textInfo(); }
	for (int i = 0; i < ru.half_spaces.size(); i++) { os << ru.half_spaces[i].textInfo(); }
	for (int i = 0; i < ru.closed_boxes.size(); i++) { os << ru.closed_boxes[i].textInfo(); }
	for (int i = 0; i < ru.boxes.size(); i++) { os << ru.boxes[i].textInfo(); }

	for (int i = 0; i < ru.closed_spheres_c.size(); i++) { os << ru.closed_spheres_c[i].textInfo(); }
	for (int i = 0; i < ru.spheres_c.size(); i++) { os << ru.spheres_c[i].textInfo(); }
	for (int i = 0; i < ru.closed_half_spaces_c.size(); i++) { os << ru.closed_half_spaces_c[i].textInfo(); }
	for (int i = 0; i < ru.half_spaces_c.size(); i++) { os << ru.half_spaces_c[i].textInfo(); }
	for (int i = 0; i < ru.closed_boxes_c.size(); i++) { os << ru.closed_boxes_c[i].textInfo(); }
	for (int i = 0; i < ru.boxes_c.size(); i++) { os << ru.boxes_c[i].textInfo(); }
	return os;
}

std::ostream& dynamic_range::operator<<(std::ostream& os, dynamic_range::rangeUnion& ru) {
	for (int i = 0; i < ru.ranges.size(); i++) { os << ru.ranges[i]->textInfo(); }
	return os;
}

namespace dynamic_range {
	template<> void rangeUnion::add_range<sphereRange_t>(const sphereRange_t& rg) { ranges.emplace_back(new sphereRange_t(rg)); }
	template<> void rangeUnion::add_range<boxRange_t>(const boxRange_t& rg) { ranges.emplace_back(new boxRange_t(rg)); }
	template<> void rangeUnion::add_range<halfSpaceRange_t>(const halfSpaceRange_t& rg) { ranges.emplace_back(new halfSpaceRange_t(rg)); }
};


namespace static_range {
	template<> void rangeUnion::dispatch_one_range(sphereRange_t<true> r) { closed_spheres.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(sphereRange_t<false> r) { spheres.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(halfSpaceRange_t<true> r) { closed_half_spaces.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(halfSpaceRange_t<false> r) { half_spaces.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(boxRange_t<true> r) { closed_boxes.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(boxRange_t<false> r) { boxes.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(cylinderRange<false> c) { cylinders.push_back(c); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<sphereRange_t<true>> r) { closed_spheres_c.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<sphereRange_t<false>> r) { spheres_c.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<halfSpaceRange_t<true>> r) { closed_half_spaces_c.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<halfSpaceRange_t<false>> r) { half_spaces_c.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<boxRange_t<true>> r) { closed_boxes_c.push_back(r); }
	template<> void rangeUnion::dispatch_one_range(complementRange_t<boxRange_t<false>> r) { boxes_c.push_back(r); }
};
