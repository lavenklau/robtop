#pragma once

#ifndef __BOUNDARY_CONDITION_GENERATOR
#define __BOUNDARY_CONDITION_GENERATOR

#include "CGAL/Simple_cartesian.h"
#include "CGAL/Bbox_3.h"
#include "CGAL/Iso_cuboid_3.h"
#include "string"
#include "sstream"
#include "Eigen/Eigen"


typedef CGAL::Simple_cartesian<double> Kernel;


namespace static_range {

	typedef double Scaler;

	template<typename range>
	struct range_t {
		constexpr static Scaler small_value = 1e-8;
		//friend typename range;
		bool contains(Scaler p[3]) {
			return static_cast<range*>(this)->contains_impl(p);
		}
		virtual ~range_t(void) {};
		std::string textInfo(void) {
			range* ptr = static_cast<range*>(this);
			std::string typeName = ptr->get_type();
			std::string Param = ptr->get_param();
			return typeName + '\n' + Param;
		}
	};

	template<bool closed = false>
	struct sphereRange_t
		:public range_t<sphereRange_t<closed>>
	{
		friend class range_t<sphereRange_t<closed>>;
		template<typename> friend class complementRange_t;
		Scaler sqrRadius;
		Scaler center[3];
		std::string get_type(void) {
			return std::string("Sphere Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) {
			std::ostringstream ostr;
			ostr << "center (" << center[0] << ", " << center[1] << ", " << center[2] << ")  ";
			ostr << "radius " << sqrt(sqrRadius) << std::endl;
			return ostr.str();
		}
		sphereRange_t(Scaler sphere_center[3], Scaler radius) :sqrRadius(radius*radius), center{ sphere_center[0],sphere_center[1],sphere_center[2] } { }
		bool contains_impl(Scaler p[3]) const {
			Scaler sqrdist = (p[0] - center[0])*(p[0] - center[0]) + (p[1] - center[1])*(p[1] - center[1]) + (p[2] - center[2])*(p[2] - center[2]);
			return sqrdist < sqrRadius || (closed && (sqrdist - sqrRadius) < range_t<sphereRange_t<closed>>::small_value);
		}
	};

	template<bool closed = false>
	struct halfSpaceRange_t
		:public range_t<halfSpaceRange_t<closed>>
	{
		Kernel::Plane_3 cutPlane;
		template<typename > friend class range_t;
		template<typename> friend class complementRange_t;
		std::string get_type(void) {
			return std::string("halfSpace Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) {
			std::ostringstream ostr;
			ostr << cutPlane << std::endl;
			return ostr.str();
		}

		halfSpaceRange_t(Scaler a, Scaler b, Scaler c, Scaler h) : cutPlane(a, b, c, h) {}
		// create half space defined by a x + b y + c z > h 

		halfSpaceRange_t(Scaler p1[3], Scaler p2[3], Scaler p3[3])
			:cutPlane(
				Kernel::Point_3(p1[0], p1[1], p1[2]),
				Kernel::Point_3(p2[0], p2[1], p2[2]),
				Kernel::Point_3(p3[0], p3[1], p3[2])
			) {}
		// create half space cut by plane through p1,p2,p3 such that p1,p2,p3 are in positive sense(counterclockwise)

		bool contains_impl(Scaler p[3]) const {
			return cutPlane.has_on_positive_side(Kernel::Point_3(p[0], p[1], p[2])) || (closed&& cutPlane.has_on(Kernel::Point_3(p[0], p[1], p[2])));
		}
	};

	template<bool closed = false>
	struct boxRange_t
		:public range_t<boxRange_t<closed>>
	{
		Kernel::Iso_cuboid_3 box;
		template<typename> friend class range_t;
		template<typename> friend class complementRange_t;
		std::string get_type(void) {
			return std::string("Box Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) {
			std::ostringstream ostr;
			ostr << '(' << box << ')' << std::endl;
			return ostr.str();
		}
		boxRange_t(Scaler p1[3], Scaler p2[3])
			:box(Kernel::Point_3(p1[0], p1[1], p1[2]), Kernel::Point_3(p2[0], p2[1], p2[2])) { }
		// create cuboid box with diagonal point p1 and p2

		boxRange_t(Scaler min_hx, Scaler min_hy, Scaler min_hz, Scaler max_hx, Scaler max_hy, Scaler max_hz, Scaler hw = Scaler{ 1 })
			:box(min_hx, min_hy, min_hz, max_hx, max_hy, max_hz, hw) {}
		// create cuboid box with opposite vertices (min_hx/hw, min_hy/hw, min_hz/hw) and (max_hx/hw, max_hy/hw, max_hz/hw).

		bool contains_impl(Scaler p[3]) const {
			//std::cout << "box range " << box << " is called " << std::endl;
			//bool contain = box.has_on_bounded_side(Kernel::Point_3(p[0], p[1], p[2]));
			//if (!contain) {
			//	Kernel::Point_3 xm = box.min();
			//	Kernel::Point_3 xM = box.max();
			//	//if ((p[0] > xm[0] && p[0] < xM[0]) && p[1] > xm[1] && p[1]<xM[1] && p[2]>xm[2] && p[2] < xM[2]) {
			//	std::cout << "xm = " << xm << std::endl;
			//	std::cout << "xM = " << xM << std::endl;
			//		std::cout << /*"box contains " <<*/ p[0] << ", " << p[1] << ", " << p[2] << std::endl;
			//	//}
			//}
			return box.has_on_bounded_side(Kernel::Point_3(p[0], p[1], p[2])) || (closed && box.has_on_boundary(Kernel::Point_3(p[0], p[1], p[2])));
		}
	};

	template<typename Range_Type>
	struct complementRange_t
		:public Range_Type
	{
		template<typename... Args>
		complementRange_t(Args... args) :Range_Type(args...) {}

		std::string textInfo(void) {
			return Range_Type::get_type() + " complement " + '\n' + Range_Type::get_param();
		}
		bool contains(Scaler p[3]) const {
			return !Range_Type::contains_impl(p);
		}
	};

	template<bool closed = false>
	struct cylinderRange
		:public range_t<cylinderRange<closed>>
	{
		Eigen::Matrix<double, 3, 1> _bottomCenter;
		Eigen::Matrix<double, 3, 1> _heightVector;
		double _radius = 0;

		cylinderRange(Eigen::Matrix<double, 3, 1> botCenter, double radii, Eigen::Matrix<double, 3, 1> vheight) 
			: _radius(radii), _bottomCenter(botCenter), _heightVector(vheight) { }

		std::string get_type(void) {
			return std::string("cylinder Range ") + (closed ? "closed" : "opened");
		}

		bool contains(Scaler p[3]) const {
			Eigen::Matrix<double, 3, 1> z = _heightVector;
			double height = z.norm();
			z = z / height;
			Eigen::Matrix<double, 3, 1> pv(p[0], p[1], p[2]);
			double proj = (pv - _bottomCenter).dot(z);
			if (proj > height || proj < 0) { return false; }
			double devi = (pv - _bottomCenter).cross(z).norm();
			if (devi >= _radius) { return false; }
			return true;
		}
	};
	

	struct rangeUnion
	{
		friend std::ostream& operator<<(std::ostream& os, rangeUnion& ru);
		std::vector<sphereRange_t<true>> closed_spheres;
		std::vector<sphereRange_t<false>> spheres;

		std::vector<halfSpaceRange_t<true>> closed_half_spaces;
		std::vector<halfSpaceRange_t<false>> half_spaces;

		std::vector<boxRange_t<true>> closed_boxes;
		std::vector<boxRange_t<false>> boxes;

		std::vector<complementRange_t<sphereRange_t<true>>> closed_spheres_c;
		std::vector<complementRange_t<sphereRange_t<false>>> spheres_c;

		std::vector<complementRange_t<halfSpaceRange_t<true>>> closed_half_spaces_c;
		std::vector<complementRange_t<halfSpaceRange_t<false>>> half_spaces_c;

		std::vector<complementRange_t<boxRange_t<true>>> closed_boxes_c;
		std::vector<complementRange_t<boxRange_t<false>>> boxes_c;

		std::vector<cylinderRange<false>> cylinders;

		template<typename Range>
		void dispatch_one_range(Range r) {
			//printf("\033[31mboundary condition generator do not support this type !\033[0m\n");
			static_assert(sizeof(Range) == 0, "boundary condition generator do not support this type !");
		}

		template<typename Range1, typename... RangeSet_Type>
		void dispatch_ranges(Range1 rg1, RangeSet_Type... ranges) {
			dispatch_one_range(rg1);
			dispatch_ranges(ranges...);
		}

		struct ParserStat {
			enum range_type_status_t {
				findingBox = 0,
				findingSphere = 1,
				findingHalfSpace = 2,
			}range_status = findingBox;
			enum complement_status_t {
				noComplement,
				Complement
			}complement_status = noComplement;
			enum closed_status_t {
				open,
				closed
			}close_status;

			// return if status changed
			bool next(const std::string& cmd) {
				if (cmd.find("box") < cmd.size()) { range_status = findingBox; return true; };
				if (cmd.find("sphere") < cmd.size()) { range_status = findingSphere; return true; }
				if (cmd.find("halfspace") < cmd.size()) { range_status = findingHalfSpace; return true; }
				if (cmd.find("closed") < cmd.size()) { close_status = closed; return true; }
				if (cmd.find("open") < cmd.size()) { close_status = open; return true; }
				if (cmd.find("complement") < cmd.size()) { complement_status = Complement; return true; }
				if (cmd.find("nocomplement") < cmd.size()) { complement_status = noComplement; return true; }
				return false;
			}
		};
	//public:

		void clear(void) {
			closed_spheres.clear();
			spheres.clear();
			closed_half_spaces.clear();
			half_spaces.clear();
			closed_boxes.clear();
			boxes.clear();
			closed_spheres_c.clear();
			spheres_c.clear();
			closed_half_spaces_c.clear();
			half_spaces_c.clear();
			closed_boxes_c.clear();
			boxes_c.clear();
		}

		void parse(std::string& s) {
			std::istringstream istr(s);
			char sbuf[512];
			std::vector<Scaler> geoparams;
			ParserStat state;
			while (istr.getline(sbuf, 512)) {
				bool status_changed = state.next(sbuf);
				if (status_changed) { geoparams.clear(); continue; };
				std::istringstream paramstr;
				Scaler var;
				paramstr = std::istringstream(sbuf);
				while (paramstr) {
					if ((paramstr >> var).fail()) { break; }
					geoparams.push_back(var);
				}
				bool closed = state.close_status == ParserStat::closed;
				bool complement = state.complement_status == ParserStat::Complement;
				switch (state.range_status) {
				case(ParserStat::findingBox):
					if (geoparams.size() == 6) {
						if (closed) {
							add_range(boxRange_t<true>(&geoparams[0], &geoparams[3]));
						}
						else {
							add_range(boxRange_t<false>(&geoparams[0], &geoparams[3]));
						}
					}
					else if (geoparams.size() == 7) {
						if (closed) {
							add_range(boxRange_t<true>(geoparams[0], geoparams[1], geoparams[2], geoparams[3], geoparams[4], geoparams[5], geoparams[6]));
						}
						else {
							add_range(boxRange_t<false>(geoparams[0], geoparams[1], geoparams[2], geoparams[3], geoparams[4], geoparams[5], geoparams[6]));
						}
					}
					break;
				case(ParserStat::findingSphere):
					if (geoparams.size() == 4) {
						if (closed) {
							add_range(sphereRange_t<true>(&geoparams[0], geoparams[3]));
						}
						else {
							add_range(sphereRange_t<false>(&geoparams[0], geoparams[3]));
						}
					}
					break;
				case(ParserStat::findingHalfSpace):
					if (geoparams.size() == 4) {
						if (closed) {
							add_range(halfSpaceRange_t<true>(geoparams[0], geoparams[1], geoparams[2], geoparams[3]));
						}
						else {
							add_range(halfSpaceRange_t<false>(geoparams[0], geoparams[1], geoparams[2], geoparams[3]));
						}
					}
					else if (geoparams.size() == 9) {
						if (closed) {
							add_range(halfSpaceRange_t<true>(&geoparams[0], &geoparams[3], &geoparams[6]));
						}
						else {
							add_range(halfSpaceRange_t<false>(&geoparams[0], &geoparams[3], &geoparams[6]));
						}
					}
					break;
				default:
					break;
				}
			}
		}

		template<typename... RangeSet_Type>
		rangeUnion(RangeSet_Type... ranges) {
			dispatch_ranges(ranges...);
		}

		rangeUnion(void) {}

		template<typename Range_Type>
		void add_range(Range_Type rg) {
			dispatch_one_range(rg);
		}

		std::function<bool(Scaler p[3])> generate(void) {
			return [=](Scaler p[3]) {
				bool is_fix = false;
				for (int i = 0; i < closed_spheres.size(); i++) { is_fix |= closed_spheres[i].contains(p); }
				for (int i = 0; i < spheres.size(); i++) { is_fix |= spheres[i].contains(p); }
				for (int i = 0; i < closed_half_spaces.size(); i++) { is_fix |= closed_half_spaces[i].contains(p); }
				for (int i = 0; i < half_spaces.size(); i++) { is_fix |= half_spaces[i].contains(p); }
				for (int i = 0; i < closed_boxes.size(); i++) { is_fix |= closed_boxes[i].contains(p); }
				for (int i = 0; i < boxes.size(); i++) { is_fix |= boxes[i].contains(p); }
				for (int i = 0; i < cylinders.size(); i++) { is_fix |= cylinders[i].contains(p); }

				for (int i = 0; i < closed_spheres_c.size(); i++) { is_fix |= closed_spheres_c[i].contains(p); }
				for (int i = 0; i < spheres_c.size(); i++) { is_fix |= spheres_c[i].contains(p); }
				for (int i = 0; i < closed_half_spaces_c.size(); i++) { is_fix |= closed_half_spaces_c[i].contains(p); }
				for (int i = 0; i < half_spaces_c.size(); i++) { is_fix |= half_spaces_c[i].contains(p); }
				for (int i = 0; i < closed_boxes_c.size(); i++) { is_fix |= closed_boxes_c[i].contains(p); }
				for (int i = 0; i < boxes_c.size(); i++) { is_fix |= boxes_c[i].contains(p); }
				return is_fix;
			};
		}

	};

	template<> void rangeUnion::dispatch_one_range(sphereRange_t<true> r);
	template<> void rangeUnion::dispatch_one_range(sphereRange_t<false> r);
	template<> void rangeUnion::dispatch_one_range(halfSpaceRange_t<true> r);
	template<> void rangeUnion::dispatch_one_range(halfSpaceRange_t<false> r);
	template<> void rangeUnion::dispatch_one_range(boxRange_t<true> r);
	template<> void rangeUnion::dispatch_one_range(boxRange_t<false> r);
	template<> void rangeUnion::dispatch_one_range(cylinderRange<false> c);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<sphereRange_t<true>> r);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<sphereRange_t<false>> r);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<halfSpaceRange_t<true>> r);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<halfSpaceRange_t<false>> r);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<boxRange_t<true>> r);
	template<> void rangeUnion::dispatch_one_range(complementRange_t<boxRange_t<false>> r);


};

namespace dynamic_range {
	typedef double Scaler;
	struct range_t {
	//protected:
		constexpr static Scaler small_value = 1e-8;
		bool closed = false;
		bool complement = false;
	//public:
		virtual bool contains(Scaler p[3]) const = 0;
		virtual std::string get_type(void) = 0;
		virtual std::string get_param(void) = 0;
		virtual std::string textInfo(void) {
			std::string typeInfo = get_type();
			std::string paramInfo = get_param();
			std::ostringstream os;
			os << "region Type : " << "<" << typeInfo << ">" << std::endl;
			os << "param : " << std::endl << paramInfo;
			return os.str();
		}
		virtual ~range_t(void) {}
		range_t(bool is_closed, bool is_complement) :closed(is_closed), complement(is_complement) {}
		range_t(void) {}
	};

	struct sphereRange_t
		: public range_t
	{
		Scaler sqrRadius;
		Scaler center[3];
	//private:
		std::string get_type(void) override {
			return std::string("Sphere Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) override {
			std::ostringstream ostr;
			ostr << "center (" << center[0] << ", " << center[1] << ", " << center[2] << ")  ";
			ostr << "radius " << sqrt(sqrRadius) << std::endl;
			return ostr.str();
		}
		bool contains(Scaler p[3]) const override {
			Scaler sqrdist = (p[0] - center[0])*(p[0] - center[0]) + (p[1] - center[1])*(p[1] - center[1]) + (p[2] - center[2])*(p[2] - center[2]);
			return (sqrdist < sqrRadius || (closed && (sqrdist - sqrRadius) < range_t::small_value)) ^ complement;
		}
	//public:
		sphereRange_t(Scaler sphere_center[3], Scaler radius) :sqrRadius(radius*radius), center{ sphere_center[0],sphere_center[1],sphere_center[2] }, range_t(){ }
		sphereRange_t(Scaler sphere_center[3], Scaler radius, bool is_closed, bool is_complement)
			:sqrRadius(radius*radius), center{ sphere_center[0],sphere_center[1],sphere_center[2] }, range_t(is_closed, is_complement){
		}

	};

	struct boxRange_t
		:public range_t
	{
	//private:
		Kernel::Iso_cuboid_3 box;
	//private:
		std::string get_type(void) override {
			return std::string("Box Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) {
			std::ostringstream ostr;
			ostr << '(' << box << ')' << std::endl;
			return ostr.str();
		}
	//public:
		boxRange_t(Scaler p1[3], Scaler p2[3])
			:box(Kernel::Point_3(p1[0], p1[1], p1[2]), Kernel::Point_3(p2[0], p2[1], p2[2])) { }
		// create cuboid box with diagonal point p1 and p2

		boxRange_t(Scaler p1[3], Scaler p2[3], bool is_closed, bool is_complement)
			:box(Kernel::Point_3(p1[0], p1[1], p1[2]), Kernel::Point_3(p2[0], p2[1], p2[2])), range_t(is_closed, is_complement) { }
		// create cuboid box with diagonal point p1 and p2

		boxRange_t(Scaler min_hx, Scaler min_hy, Scaler min_hz, Scaler max_hx, Scaler max_hy, Scaler max_hz, Scaler hw = Scaler{ 1 })
			:box(min_hx, min_hy, min_hz, max_hx, max_hy, max_hz, hw) {}
		// create cuboid box with opposite vertices (min_hx/hw, min_hy/hw, min_hz/hw) and (max_hx/hw, max_hy/hw, max_hz/hw).

		boxRange_t(Scaler min_hx, Scaler min_hy, Scaler min_hz, Scaler max_hx, Scaler max_hy, Scaler max_hz, Scaler hw, bool is_closed, bool is_complement)
			:box(min_hx, min_hy, min_hz, max_hx, max_hy, max_hz, hw), range_t(is_closed, is_complement) {}
		// create cuboid box with opposite vertices (min_hx/hw, min_hy/hw, min_hz/hw) and (max_hx/hw, max_hy/hw, max_hz/hw).

		bool contains(Scaler p[3]) const override {
			return (box.has_on_bounded_side(Kernel::Point_3(p[0], p[1], p[2])) || (closed && box.has_on_boundary(Kernel::Point_3(p[0], p[1], p[2])))) ^ complement;
		}
	};

	struct halfSpaceRange_t
		:public range_t
	{
	//private:
		Kernel::Plane_3 cutPlane;
	//private:
		std::string get_type(void) {
			return std::string("halfSpace Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) {
			std::ostringstream ostr;
			ostr << cutPlane;
			return ostr.str();
		}

	//public:
		halfSpaceRange_t(Scaler a, Scaler b, Scaler c, Scaler h) : cutPlane(a, b, c, h) {}
		// create half space defined by a x + b y + c z > h 

		halfSpaceRange_t(Scaler a, Scaler b, Scaler c, Scaler h, bool is_closed, bool is_complement) : cutPlane(a, b, c, h), range_t(is_closed, is_complement) {}
		// create half space defined by a x + b y + c z > h 

		halfSpaceRange_t(Scaler p1[3], Scaler p2[3], Scaler p3[3])
			:cutPlane(
				Kernel::Point_3(p1[0], p1[1], p1[2]),
				Kernel::Point_3(p2[0], p2[1], p2[2]),
				Kernel::Point_3(p3[0], p3[1], p3[2])
			) {}
		// create half space cut by plane through p1,p2,p3 such that p1,p2,p3 are in positive sense(counterclockwise)

		halfSpaceRange_t(Scaler p1[3], Scaler p2[3], Scaler p3[3], bool is_closed, bool is_complement)
			:cutPlane(
				Kernel::Point_3(p1[0], p1[1], p1[2]),
				Kernel::Point_3(p2[0], p2[1], p2[2]),
				Kernel::Point_3(p3[0], p3[1], p3[2])
			),
			range_t(is_closed, is_complement) {}
		// create half space cut by plane through p1,p2,p3 such that p1,p2,p3 are in positive sense(counterclockwise)

		bool contains(Scaler p[3]) const override {
			return (cutPlane.has_on_positive_side(Kernel::Point_3(p[0], p[1], p[2])) || (closed&& cutPlane.has_on(Kernel::Point_3(p[0], p[1], p[2])))) ^ complement;
		}
	};


	struct  cylinderRange_t
		: public range_t
	{
		Eigen::Matrix<double, 3, 1> _bottomCenter;
		Eigen::Matrix<double, 3, 1> _heightVector;
		double _radius = 0;
		cylinderRange_t(Eigen::Matrix<double, 3, 1> botCenter, double radii, Eigen::Matrix<double, 3, 1> vheight)
			: _radius(radii), _bottomCenter(botCenter), _heightVector(vheight) { }

		bool contains(Scaler p[3]) const override {
			Eigen::Matrix<double, 3, 1> z = _heightVector;
			double height = z.norm();
			z = z / height;
			Eigen::Matrix<double, 3, 1> pv(p[0], p[1], p[2]);
			double proj = (pv - _bottomCenter).dot(z);
			if (proj > height || proj < 0) { return false; }
			double devi = (pv - _bottomCenter).cross(z).norm();
			if (devi >= _radius) { return false; }
			return true;
		}

		std::string get_type(void) override {
			return std::string("halfSpace Range ") + (closed ? "closed" : "opened");
		}
		std::string get_param(void) override {
			std::ostringstream ostr;
			ostr << _bottomCenter << _heightVector << _radius;
			return ostr.str();
		}
	};

	struct rangeUnion {
	//private:
		std::vector<range_t*> ranges;
		friend std::ostream& operator<<(std::ostream& os, dynamic_range::rangeUnion& ru);
	//public:
		~rangeUnion() {}
		std::function<bool(Scaler[3])> generate(void) {
			return [=](Scaler p[3]) {
				bool is_in_region = false;
				for (int i = 0; i < ranges.size(); i++) {
					is_in_region |= ranges[i]->contains(p);
				}
				return is_in_region;
			};
		}

	//public:
		template<typename range>
		void add_range(const range& rg) {
			static_assert(sizeof(range) == 0, "undefined range !");
		}

	};

	template<> void rangeUnion::add_range<sphereRange_t>(const sphereRange_t& rg);
	template<> void rangeUnion::add_range<boxRange_t>(const boxRange_t& rg);
	template<> void rangeUnion::add_range<halfSpaceRange_t>(const halfSpaceRange_t& rg);



	class constant_normGen_t {
		Scaler constNorm;
	public:
		Scaler gen(Scaler p[3])const { return constNorm; }
		constant_normGen_t(Scaler c) :constNorm(c) {}
	};

	class constant_dirGen_t {
		Eigen::Matrix<Scaler, 3, 1> const_direction;
	public:
		Eigen::Matrix<Scaler, 3, 1> gen(Scaler p[3]) const { return const_direction; }
		constant_dirGen_t(const Eigen::Matrix<Scaler, 3, 1>& dir) : const_direction(dir.normalized()) {}
	};

	class radial_normGen_t {
		Eigen::Matrix<Scaler, 3, 1> radialCenter;
		Scaler damp_ratio;
		Scaler amplifier;
	public:
		Scaler gen(Scaler p[3]) const {
			Scaler radi = (radialCenter - Eigen::Matrix<Scaler, 3, 1>::Map(p)).norm();
			return amplifier * exp(-pow(radi, damp_ratio * 2));
		}
		radial_normGen_t(const Eigen::Matrix<Scaler, 3, 1>& rc, Scaler amplifier_ = 1, Scaler damp_ratio_ = 1)
			: radialCenter(rc), damp_ratio(damp_ratio_), amplifier(amplifier_) { }
	};

	class radial_dirGen_t {
		Eigen::Matrix<Scaler, 3, 1> radialCenter;
	public:
		Eigen::Matrix<Scaler, 3, 1> gen(Scaler p[3]) const {
			return (radialCenter - Eigen::Matrix<Scaler, 3, 1>::Map(p)).normalized();
		}
		radial_dirGen_t(const Eigen::Matrix<Scaler, 3, 1>& rc)
			:radialCenter(rc) {}
	};

	class twist_dirGen_t {
		Eigen::Matrix<Scaler, 3, 1> twistAxisStart;
		Eigen::Matrix<Scaler, 3, 1> twistAxisEnd;
	public:
		Eigen::Matrix<Scaler, 3, 1> gen(Scaler p[3]) const {
			return (Eigen::Matrix<Scaler, 3, 1>::Map(p) - twistAxisStart).cross(twistAxisEnd - twistAxisStart).normalized();
		}
		explicit twist_dirGen_t(const Eigen::Matrix<Scaler, 3, 1>& axis_a, const Eigen::Matrix<Scaler, 3, 1>& axis_b)
			: twistAxisStart(axis_a), twistAxisEnd(axis_b) { }
	};

	class twist_normGen_t {
		Eigen::Matrix<Scaler, 3, 1> twistAxisStart;
		Eigen::Matrix<Scaler, 3, 1> twistAxisEnd;
		double ampl;
	public:
		Scaler gen(Scaler p[3]) const {
			Eigen::Matrix<Scaler, 3, 1> vp(p[0], p[1], p[2]);
			return (vp - twistAxisStart).cross((twistAxisEnd - twistAxisStart).normalized()).norm() * ampl;
		}
		explicit twist_normGen_t(const Eigen::Matrix<Scaler, 3, 1>& axis_a, const Eigen::Matrix<Scaler, 3, 1>& axis_b, double amplifier)
			: twistAxisEnd(axis_b), twistAxisStart(axis_a), ampl(amplifier) { }
	};

	template<typename normGen_t>
	struct is_normGenrator { static constexpr bool value = false; };

	template<> struct is_normGenrator<constant_normGen_t> { static constexpr bool value = true; };
	template<> struct is_normGenrator<radial_normGen_t> { static constexpr bool value = true; };
	template<> struct is_normGenrator<twist_normGen_t> { static constexpr bool value = true; };

	template<typename dirGen_t>
	struct is_directionGenrator { static constexpr bool value = false; };

	template<> struct is_directionGenrator<constant_dirGen_t> { static constexpr bool value = true; };
	template<> struct is_directionGenrator<radial_dirGen_t> { static constexpr bool value = true; };
	template<> struct is_directionGenrator<twist_dirGen_t> { static constexpr bool value = true; };

	class vectorFieldBase_t
	{
	public:
		typedef Eigen::Matrix<Scaler, 3, 1> vec3x;
		virtual vec3x at(Scaler p[3]) const = 0;
		virtual ~vectorFieldBase_t(void) {}
	};

	template<typename normGen_t, typename dirGen_t>
	class vectorField_t
		: public vectorFieldBase_t,
		private std::enable_if<is_normGenrator<normGen_t>::value, normGen_t>::type,
		private std::enable_if<is_directionGenrator<dirGen_t>::value, dirGen_t>::type
	{
	public:
		vectorField_t(const normGen_t& ng, const dirGen_t& dg)
			:normGen_t(ng), dirGen_t(dg) {}

		virtual vec3x at(Scaler p[3]) const override {
			return normGen_t::gen(p)*dirGen_t::gen(p);
		};
	};

	typedef vectorField_t<constant_normGen_t, constant_dirGen_t> constant_vectorField_t;
	typedef constant_vectorField_t cvField_t;
	typedef vectorField_t<radial_normGen_t, constant_dirGen_t> radialNormConstDirField_t;
	typedef radialNormConstDirField_t rncdField_t;
	typedef vectorField_t<constant_normGen_t, radial_dirGen_t> constNormRadialDirField_t;
	typedef constNormRadialDirField_t cnrdField_t;
	typedef vectorField_t<radial_normGen_t, radial_dirGen_t> radialNormRadialDirField_t;
	typedef radialNormRadialDirField_t rnrdField_t;
	typedef vectorField_t<twist_normGen_t, twist_dirGen_t> twistNormTwistDirField_t;
	typedef twistNormTwistDirField_t tntdField_t;


	class rangeField_t
	{
	private:
		range_t* rg;
		vectorFieldBase_t* vec;
	public:
		~rangeField_t() {
			delete rg;
			delete vec;
		}
		Eigen::Matrix<Scaler, 3, 1> at(Scaler p[3]) {
			if (rg->contains(p)) {
				return vec->at(p);
			}
			else {
				return Eigen::Matrix<Scaler, 3, 1>::Zero();
			}
		}
		rangeField_t(range_t* field_range_, vectorFieldBase_t* fieldValue_)
			: rg(field_range_), vec(fieldValue_) {}
	};

	class Field_t {
	private:
		std::vector<rangeField_t*> rgFields;
		typedef Eigen::Matrix<Scaler, 3, 1> vec3x;
	public:
		~Field_t(void) {
			for (int i = 0; i < rgFields.size(); i++) {
				delete rgFields[i];
			}
		}
		Eigen::Matrix<Scaler, 3, 1> at(Scaler p[3]) {
			Eigen::Matrix<Scaler, 3, 1> vsum;
			vsum.fill(0);
			for (int i = 0; i < rgFields.size(); i++) {
				auto v = rgFields[i]->at(p);
				vsum += v;
			}
			return vsum;
		}
		void add_field(range_t* rg, vectorFieldBase_t* vec) {
			rgFields.emplace_back(new rangeField_t(rg, vec));
		}
		std::function<vec3x(Scaler[3])> generate(void) {
			return [=](Scaler p[3]) {
				return this->at(p);
			};
		}
		void clear(void) {
			for (int i = 0; i < rgFields.size(); i++) {
				delete rgFields[i];
			}
			rgFields.clear();
		}
	};
};

std::ostream& static_range::operator<<(std::ostream& os, static_range::rangeUnion& ru);

std::ostream& dynamic_range::operator<<(std::ostream& os, dynamic_range::rangeUnion& ru);

#endif


