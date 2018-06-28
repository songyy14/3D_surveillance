#include <Windows.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

#include <thread>
#include <chrono>
#include <mutex>

#include <osgViewer/Viewer>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Point>
#include <osg/Texture2D>
#include <osgDB/Readfile>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;


// Encapsulate methods of generating data and reading data.
class DataSupplier {
public:
	static void data_generator() {
		
		// Time to ajust the view.
		this_thread::sleep_for(chrono::seconds(30));

		int num_cams = 2;
		string cams[] = {"cam3", "cam4"};
		string dir = "C:/E/Matlab/Object Tracking/dataset/";

		vector<MatrixXd> camera_matrices(num_cams);
		vector<fstream> files(num_cams);

		for (int k = 0; k < num_cams; k++) {
			camera_matrices[k] = read_camera_matrix(dir + cams[k] + "/camera_matrix_our.txt");
			files[k].open(dir + cams[k] + "/track.txt", fstream::in);
		}

		string tmp;
		int num_trackers;
		int frame_id, pid, x, y, w, h;
		char filename[8];

		Mat frame;
		vector<osg::ref_ptr<osg::Image>> _images(0);
		vector<osg::ref_ptr<osg::Vec3Array>> _vertices(0);

		vector<int> pids(0);

		while (true) {

			// Terminate this worker thread when the program exits.
			mtx2.lock();
			if (to_terminate)
				break;
			mtx2.unlock();

			_images.clear();
			_vertices.clear();
			pids.clear();

			for (int k = 0; k < num_cams; k++) {

				// Exit loop when reaching end of file.
				frame_id = -1;
				files[k] >> tmp >> frame_id;
				if (frame_id == -1 || frame_id == 1000) {
					to_terminate = true;
					break;
				}

				cout << cams[k] << "  #frame " << frame_id << endl;

				// Read current frame as the texture.
				sprintf_s(filename, "%04d", frame_id);
				frame = imread(dir + cams[k] + "/img/" + filename + ".jpg");

				// Compute vertices array and texture image for each person quad.
				files[k] >> num_trackers;
				for (int i = 0; i < num_trackers; i++) {

					files[k] >> pid >> x >> y >> w >> h;
					// Only render one quad for each person,
					// even if he shows up in multiple cameras.
					bool show_up = false;
					for (int j = 0; j < pids.size(); j++) {
						if (pids[j] == pid) {
							show_up = true;
							break;
						}
					}
					if (show_up)
						continue;
					else
						pids.push_back(pid);

					correct_coord(x, y, w, h, frame.cols, frame.rows);

					// Crop the image region containing the tracked persons.
					Mat region = frame(Rect(x, y, w, h));
					_images.push_back(mat2OsgImage(region));

					// Vertices array.
					_vertices.push_back(new osg::Vec3Array);
					back_projection(camera_matrices[k], x, y, w, h, _vertices[_vertices.size() - 1]);
				}
			}

			if (!to_terminate) {
				mtx1.lock();
				data_ready = true;
				clone(_images, images);
				clone(_vertices, vertices);
				mtx1.unlock();
			}
		}

		cout << "data_generator quit" << endl;
		
	}

	static bool data_reader(vector<osg::ref_ptr<osg::Image>>& images0, vector<osg::ref_ptr<osg::Vec3Array>>& vertices0) {

		bool return_val;

		mtx1.lock();
		if (!data_ready)
			return_val = false;
		else {
			return_val = true;
			data_ready = false;
			clone(images, images0);
			clone(vertices, vertices0);
		}
		mtx1.unlock();

		return return_val;
	}

	static void terminate() {
		mtx2.lock();
		to_terminate = true;
		mtx2.unlock();
	}

private:
	static bool to_terminate;
	static bool data_ready;
	static vector<osg::ref_ptr<osg::Image>> images;
	static vector<osg::ref_ptr<osg::Vec3Array>> vertices;

	static mutex mtx1;
	static mutex mtx2;

	static void correct_coord(int& x, int& y, int& w, int& h, int cols, int rows) {
		// For columns.
		if (x < 0)
			x = 0;
		if (x + w > cols)
			w = cols - x;

		// For rows.
		if (y < 0)
			y = 0;
		if (y + h > rows)
			h = rows - y;
	}

	static osg::ref_ptr<osg::Image> mat2OsgImage(cv::Mat &mat) {
		// Assume mat is a RGB image.

		int rows = mat.rows;
		int cols = mat.cols;

		// Construct an osg::Image object.
		osg::ref_ptr<osg::Image> result = new osg::Image;
		result->setOrigin(osg::Image::Origin::TOP_LEFT);
		result->allocateImage(cols, rows, 1, GL_RGB, GL_UNSIGNED_BYTE);
		unsigned char *ptr = result->data();

		// Assign pixel values to the object.
		cv::MatIterator_<cv::Vec3b> it;
		for (it = mat.begin<cv::Vec3b>(); it != mat.end<cv::Vec3b>(); it++) {
			ptr[0] = (*it)[2];
			ptr[1] = (*it)[1];
			ptr[2] = (*it)[0];
			ptr = ptr + 3;
		}

		return result.release();
	}

	static void clone(vector<osg::ref_ptr<osg::Image>>& src, vector<osg::ref_ptr<osg::Image>>& dst) {
		
		dst.resize(src.size());
		for (int i = 0; i < src.size(); i++) {
			dst[i] = static_cast<osg::Image *>(src[i]->clone(osg::CopyOp::DEEP_COPY_ALL));
		}
	}

	static void clone(vector<osg::ref_ptr<osg::Vec3Array>>& src, vector<osg::ref_ptr<osg::Vec3Array>>& dst) {

		dst.resize(src.size());
		for (int i = 0; i < src.size(); i++) {
			dst[i] = static_cast<osg::Vec3Array *>(src[i]->clone(osg::CopyOp::DEEP_COPY_ALL));
		}
	}
};


// Definition of static member variables for class `DataSupplier`.
bool DataSupplier::to_terminate = false;
bool DataSupplier::data_ready = false;
vector<osg::ref_ptr<osg::Image>> DataSupplier::images;
vector<osg::ref_ptr<osg::Vec3Array>> DataSupplier::vertices;
mutex DataSupplier::mtx1;
mutex DataSupplier::mtx2;


// Callback class for a OSG node,
// of which `operator()` method is called when the node is updated.
class QuadsNodeCallback : public osg::NodeCallback {
public:

	QuadsNodeCallback() {
		count = 0;
		start_count = getCPUTickCount();
		count_data = 0;

		images = vector<osg::ref_ptr<osg::Image>>(0);
		vertices = vector<osg::ref_ptr<osg::Vec3Array>>(0);

		texcoords = new osg::Vec2Array;
		texcoords->push_back(osg::Vec2(0, 0));
		texcoords->push_back(osg::Vec2(1, 0));
		texcoords->push_back(osg::Vec2(1, 1));
		texcoords->push_back(osg::Vec2(0, 1));
	}

	virtual void operator() (osg::Node *node, osg::NodeVisitor *nv) {

		// FPS of display and frequency of data updating.
		count++;
		double time = (getCPUTickCount() - start_count) / (double)getTickFrequency();
		if (time >= 2) {
			cout << endl << count / time << " frames per second" << endl;
			cout << count_data / time << " new data per second" << endl << endl;
		}

		data_ready = DataSupplier::data_reader(images, vertices);

		if (data_ready) {
			
			count_data++;

			// Remove all exsited quads and add new from scratch.
			osg::Geode *pq_node = static_cast<osg::Geode *>(node);
			pq_node->removeDrawables(0, pq_node->getNumDrawables());
			for (int i = 0; i < images.size(); i++) {
				osg::ref_ptr<osg::Geometry> new_quad = new osg::Geometry;

				new_quad->setVertexArray(vertices[i]);
				new_quad->setTexCoordArray(0, texcoords);
				new_quad->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

				osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
				texture->setImage(images[i]);
				new_quad->getOrCreateStateSet()->setTextureAttributeAndModes(0, texture);
				
				pq_node->addDrawable(new_quad);
			}
			
		}

		traverse(node, nv);
	}

private:
	bool data_ready;
	vector<osg::ref_ptr<osg::Image>> images;
	vector<osg::ref_ptr<osg::Vec3Array>> vertices;
	osg::ref_ptr<osg::Vec2Array> texcoords;

	int count;
	double start_count;
	int count_data;
};


void rendering() {
	// Create point cloud node (sparse).
	//string file_points = "C:/E/3d reconstruction/square_700/model_text/points3D.txt";
	//float point_size = 0.1;
	//osg::ref_ptr<osg::Geometry> point_cloud = create_point_cloud(file_points, point_size);
	//osg::ref_ptr<osg::Geode> pc_node = new osg::Geode;
	//pc_node->addDrawable(point_cloud);

	// Create point cloud node (dense).
	osg::ref_ptr<osg::Node> pc_node = osgDB::readNodeFile("C:/E/3d reconstruction/square_700/fused.ply");
	pc_node->setDataVariance(osg::Object::STATIC);

	// Create person quad node.
	osg::ref_ptr<osg::Geode> pq_node = new osg::Geode;
	pq_node->setUpdateCallback(new QuadsNodeCallback);
	pq_node->setDataVariance(osg::Object::DYNAMIC);

	// Root node contains point cloud node and person quad node.
	osg::ref_ptr<osg::Group> root = new osg::Group;
	root->addChild(pq_node);
	//root->addChild(create_ground());
	root->addChild(pc_node);
	root->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

	osgViewer::Viewer viewer;
	viewer.getCamera()->setClearColor(osg::Vec4(1, 1, 1, 1));
	viewer.setSceneData(root);
	viewer.setUpViewInWindow(200, 200, 600, 600);
	viewer.run();
}


int main() {

	// Start a worker thread to generate data.
	thread worker(DataSupplier::data_generator);

	// Do scene rendering in main thread.
	rendering();

	// Terminate worker thread when rendering window is closed.
	DataSupplier::terminate();
	worker.join();

	return 0;
}