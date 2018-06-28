from cvt cimport *
from libcpp cimport bool

cdef extern from "C:/E/C++/Tracking/KCF_test/kcftracker.hpp":
	cdef cppclass KCFTracker:
		KCFTracker(int, bool, bool, bool, bool)
		void init(Rect, Mat)
		Rect update(Mat)
		Rect get_roi()

		# attributes
		bool out_of_sight
		bool model_drift
		bool occluded_so_long
		int pid
		int occluded_time


cdef class kcftracker:
	cdef KCFTracker *classptr
	
	def __cinit__(self, pid, hog, fixed_window, multiscale, lab):
		self.classptr = new KCFTracker(pid, hog, fixed_window, multiscale, lab)
		
	def __dealloc(self):
		del self.classptr
		
	def init(self, rectlist, ary):
		self.classptr.init(pylist2cvrect(rectlist), nparray2cvmat(ary))
		
	def update(self, ary):
		rect = self.classptr.update(nparray2cvmat(ary))
		return cvrect2pylist(rect)

	def get_roi(self):
		return cvrect2pylist(self.classptr.get_roi())

	@property
	def out_of_sight(self):
		return self.classptr.out_of_sight

	@property
	def model_drift(self):
		return self.classptr.model_drift

	@property
	def occluded_so_long(self):
		return self.classptr.occluded_so_long

	@property
	def pid(self):
		return self.classptr.pid

	@property
	def occluded_time(self):
		return self.classptr.occluded_time