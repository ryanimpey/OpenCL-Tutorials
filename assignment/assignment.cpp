#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

// Returns console information about different flags that can be passed to the function
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void perform_colour_op(CImg<unsigned char>, int, int);
void perform_greyscale_op(CImg<unsigned char>, int, int);

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	// Define platform and device ID's as defaults incase no arguements are passed through
	int platform_id = 0;
	int device_id = 0;

	// Load in our initial reference file
	string inputImgFilename = "test.ppm";

	// Handle command line arguements
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { inputImgFilename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	// Hides CImg library messages/exceptions from the output
	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		// Returns a pointer to a image location from its filename
		CImg<unsigned char> inputImgPtr(inputImgFilename.c_str());
		bool IS_COLOUR = inputImgPtr.spectrum() == 3;

		// Report image width, height, and pixel count
		cout << "==============================\n" << "Results for " << inputImgFilename << "\n==============================" << endl;
		cout << "[INFO] Image Width: " << inputImgPtr.width() << ", Height: " << inputImgPtr.height() << ", Pixel Count: " << inputImgPtr.height() * inputImgPtr.width() << endl;
		cout << "[INFO] Image is ";

		if (IS_COLOUR) {
			cout << "colour (Spectrum value of 3)." << endl;
			perform_colour_op(inputImgPtr, platform_id, device_id);
		}
		else {
			cout << "greyscale (Spectrum value of 1)." << endl;
			perform_greyscale_op(inputImgPtr, platform_id, device_id);
		}
	}
	catch (const cl::Error& err) {
		// Handle any exceptions that occur during build/runtime
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		// Handle any CImg related exceptions that occur during build/runtime
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	// Return 0 to terminate the application
	return 0;
}

// Performs contrast adjustment for a colour image
void perform_colour_op(CImg<unsigned char> inputImgPtr, int platform_id, int device_id) {
	// Select platform and device to use to create a context from
	cl::Context context = GetContext(platform_id, device_id);

	// Display the selected device
	cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

	// Create a queue to which we will push commands for the device & enable profiling
	cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

	// Create program source object to reference kernel files
	cl::Program::Sources sources;
	AddSources(sources, "kernels/assign_kernels.cl");

	// Create a program to combine context and kernels
	cl::Program program(context, sources);

	// Attempt to build the OpenCL Program and catch any errors that occur during build
	try {
		program.build();
	}
	catch (const cl::Error& err) {
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		// Throw error and exit
		throw err;
		return;
	}

	// Get info about the device we're operating on
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];

	const int BIN_SIZE = 256; // Hard-coded bin size of 256
	const size_t HIST_SIZE = BIN_SIZE * sizeof(int); // Hard-coded bin size
	cl::Event inputProf, outputProf; // Create generic CL Events for profiling

	/* PART 1 - Histogram Generation [COLOUR] */
	std::vector<int> rHistBin(BIN_SIZE), gHistBin(BIN_SIZE), bHistBin(BIN_SIZE); // Create a histogram for RGB individually

	// Create our initial buffers for usage in OpenCL Kernels
	cl::Buffer inputImgBuffer(context, CL_MEM_READ_ONLY, inputImgPtr.size()); // Create a read-only buffer with a size of our input image
	cl::Buffer histBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE);  // Create a read-write buffer with the size of our histogram bin (bin_size * size(int))
	cl::Buffer channelBuffer(context, CL_MEM_READ_ONLY, sizeof(int)); // Create buffer to store current channel

	// Write image input data to our device's memory via our image input buffer
	queue.enqueueWriteBuffer(inputImgBuffer, CL_TRUE, 0, inputImgPtr.size(), &inputImgPtr.data()[0], NULL, &inputProf);

	// Load Histogram RGB Kernel
	cl::Kernel kernelHist = cl::Kernel(program, "histogram_rgb"); // Load the histogram kernel defined in my_kernels

	// Report stats for histogram kernel
	cout << "[Part 1] Maximum Work Group Size: ";
	cerr << kernelHist.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 1] Preferred Work Group Size: ";
	cerr << kernelHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Report image buffer write time
	cout << "[Part 1] Image Buffer Memory Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;


	// Event for tracking kernel execution time
	cl::Event histogramProf;

	// Execute the histogram_rgb for each image channel individually
	for (int channel = 0 ; channel < 3; channel++) {
		queue.enqueueFillBuffer(histBuffer, 0, 0, HIST_SIZE); // Fill histogram buffer with 0's
		queue.enqueueWriteBuffer(channelBuffer, CL_TRUE, 0, sizeof(int), &channel); // Write channel value to channel buffer

		// Set kernel arguements for histogram_rgb
		kernelHist.setArg(0, inputImgBuffer);
		kernelHist.setArg(1, histBuffer);
		kernelHist.setArg(2, channelBuffer);

		// Execute the kernel with our provided params
		queue.enqueueNDRangeKernel(kernelHist, cl::NullRange, cl::NDRange(inputImgPtr.size()), kernelHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device), NULL, &histogramProf);

		// Write the histogram result from our device memory to our vector via the histogram buffer
		if (channel == 0) {
			queue.enqueueReadBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &rHistBin[0], NULL, &outputProf);
		}
		else if (channel == 1) {
			queue.enqueueReadBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &gHistBin[0], NULL, &outputProf);
		}
		else {
			queue.enqueueReadBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &bHistBin[0], NULL, &outputProf);
		}

		cout << "[Part 1] [Channel "<< channel << "] Histogram Buffer Memory Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "[Part 1] [Channel " << channel << "] Histogram Kernel Execution Time [ns]:" << histogramProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogramProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "[Part 1] [Channel " << channel << "] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(histogramProf, ProfilingResolution::PROF_NS) << endl;

	}

	/* PART 2 - Cumulative Histogram Generation [COLOUR] */
	std::vector<int> rCumHist(BIN_SIZE), gCumHist(BIN_SIZE), bCumHist(BIN_SIZE); // Create 3 vectors to store cumulative R,G,B histogram values

	cl::Kernel kernelCum = cl::Kernel(program, "scan_add_atomic"); // Load Scanning kernel
	cl::Buffer cumHistBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE); // Create buffer to store cumulative values

	// Report stats for histogram kernel
	cout << "[Part 1] Maximum Work Group Size: ";
	cerr << kernelCum.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 1] Preferred Work Group Size: ";
	cerr << kernelCum.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event cumulativeProf;

	// Execute scan_add_atomic for each spectrum (r,g,b) sequentially
	for (int i = 0; i < 3; i++) {
		queue.enqueueFillBuffer(cumHistBuffer, 0, 0, HIST_SIZE); // Fill cumulative buffer with 0's

		// Queue a write of the correct histgram buffer
		switch (i) {
			case 0:
				queue.enqueueWriteBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &rHistBin[0], NULL, &inputProf);
				break;
			case 1:
				queue.enqueueWriteBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &gHistBin[0], NULL, &inputProf);
				break;
			case 2:
			default:
				queue.enqueueWriteBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &bHistBin[0], NULL, &inputProf);
				break;
		}

		cout << "[Part 2] [Channel " << i << "] Histogram Buffer Memory Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		// Set kernel arguements for scanning kernel
		kernelCum.setArg(0, histBuffer);
		kernelCum.setArg(1, cumHistBuffer);

		// Execute the cumulative histogram kernel on the selected device
		queue.enqueueNDRangeKernel(kernelCum, cl::NullRange, cl::NDRange(rHistBin.size()), kernelCum.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device), NULL, &cumulativeProf);

		// Queue a read of the correct histgram buffer
		switch (i) {
		case 0:
			queue.enqueueReadBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &rCumHist[0], NULL, &outputProf);
			break;
		case 1:
			queue.enqueueReadBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &gCumHist[0], NULL, &outputProf);
			break;
		case 2:
		default:
			queue.enqueueReadBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &bCumHist[0], NULL, &outputProf);
			break;
		}

		cout << "[Part 2] [Channel " << i << "] Cumulative Buffer Read Execution Time [ns]:" << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "[Part 2] [Channel " << i << "] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(cumulativeProf, ProfilingResolution::PROF_NS) << endl;
		cout << "[Part 2] [Channel " << i << "] Cumulative Kernel Execution Time [ns]:" << cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}

	/* PART 3 - Normalise Histogram */
	std::vector<int> rNormHist(BIN_SIZE), gNormHist(BIN_SIZE), bNormHist(BIN_SIZE); // Create 3 vectors to store normalised R,G,B histogram values

	cl::Kernel kernelNormHist = cl::Kernel(program, "norm_bins"); // Load the norm_bins kernel defined in my_kernels
	cl::Buffer normHistBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE); // Create buffer to store normalised histogram
	cl::Buffer pixelCountBuffer(context, CL_MEM_READ_ONLY, sizeof(float)); // Create buffer to store normalisation calc

	// Report stats for histogram kernel
	cout << "[Part 1] Maximum Work Group Size: ";
	cerr << kernelNormHist.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 1] Preferred Work Group Size: ";
	cerr << kernelNormHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event normalisedProf;
	
	float pixelCount = (float)255 / (float)(inputImgPtr.height() * inputImgPtr.width()); // Obtain pixel count of image
	queue.enqueueWriteBuffer(pixelCountBuffer, CL_TRUE, 0, sizeof(int), &pixelCount); // Write pixel count value to buffer

	// Execute norm_bins for each spectrum (r,g,b) sequentially
	for (int i = 0; i < 3; i++) {
		queue.enqueueFillBuffer(normHistBuffer, 0, 0, HIST_SIZE); // Fill normalised buffer with 0's
		
		// Queue a write of the correct histgram buffer
		switch (i) {
		case 0:
			queue.enqueueWriteBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &rCumHist[0], NULL, &inputProf);
			break;
		case 1:
			queue.enqueueWriteBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &gCumHist[0], NULL, &inputProf);
			break;
		case 2:
		default:
			queue.enqueueWriteBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &bCumHist[0], NULL, &inputProf);
			break;
		}

		cout << "[Part 3] [Channel " << i << "] Cumulative Buffer Memory Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		kernelNormHist.setArg(0, cumHistBuffer); // Load in the cumulative histogram buffer
		kernelNormHist.setArg(1, normHistBuffer); // Pass in our normalised buffer filled with 0's
		kernelNormHist.setArg(2, pixelCountBuffer); // Pass in the pixel count

		// Execute the cumulative histogram kernel on the selected device
		queue.enqueueNDRangeKernel(kernelNormHist, cl::NullRange, cl::NDRange(rHistBin.size()), kernelNormHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device));

		// Queue a read of the correct histgram buffer
		switch (i) {
		case 0:
			queue.enqueueReadBuffer(normHistBuffer, CL_TRUE, 0, HIST_SIZE, &rNormHist[0]);
			break;
		case 1:
			queue.enqueueReadBuffer(normHistBuffer, CL_TRUE, 0, HIST_SIZE, &gNormHist[0]);
			break;
		case 2:
		default:
			queue.enqueueReadBuffer(normHistBuffer, CL_TRUE, 0, HIST_SIZE, &bNormHist[0]);
			break;
		}

		cout << "[Part 3] [Channel " << i << "] Normalised Buffer Read Execution Time [ns]:" << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "[Part 3] [Channel " << i << "] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(cumulativeProf, ProfilingResolution::PROF_NS) << endl;
		cout << "[Part 3] [Channel " << i << "] Normalise Kernel Execution Time [ns]:" << cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}

	/* PART 4 - LOOK UP TABLE & OUTPUT */
	// Create an output buffer to store values copied from device once computation is complete
	vector<unsigned char> outputImgVect(inputImgPtr.size());
	// Create a new buffer to hold data about our output image
	cl::Buffer outputImgBuffer(context, CL_MEM_READ_WRITE, inputImgPtr.size()); //should be the same as input image

	// Create output buffers for RGB normalised values & write normalised values to each buffer
	cl::Buffer rOutBuffer(context, CL_MEM_READ_ONLY, HIST_SIZE), gOutBuffer(context, CL_MEM_READ_ONLY, HIST_SIZE), bOutBuffer(context, CL_MEM_READ_ONLY, HIST_SIZE);
	queue.enqueueWriteBuffer(rOutBuffer, CL_TRUE, 0, HIST_SIZE, &rNormHist[0]);
	queue.enqueueWriteBuffer(gOutBuffer, CL_TRUE, 0, HIST_SIZE, &gNormHist[0]);
	queue.enqueueWriteBuffer(bOutBuffer, CL_TRUE, 0, HIST_SIZE, &bNormHist[0]);


	cl::Kernel kernelLut = cl::Kernel(program, "lut_rgb"); // Load the LUT kernel defined in my_kernels
	kernelLut.setArg(0, inputImgBuffer); // Load in our normalised histogram buffer bin
	kernelLut.setArg(1, outputImgBuffer); // Load in our input image in buffer form
	kernelLut.setArg(2, rOutBuffer); // Load in our output image buffer for writing to
	kernelLut.setArg(3, gOutBuffer); // Load in our output image buffer for writing to
	kernelLut.setArg(4, bOutBuffer); // Load in our output image buffer for writing to

	// Report stats for normalisation kernel
	cout << "[Part 4] Maximum Work Group Size: ";
	cerr << kernelLut.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 4] Preferred Work Group Size: ";
	cerr << kernelLut.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event lutProf;


	// Execute lut_rgb kernel
	queue.enqueueNDRangeKernel(kernelLut, cl::NullRange, cl::NDRange(inputImgPtr.size()), cl::NDRange(256), NULL, &lutProf);

	// Copy the result from device to host
	queue.enqueueReadBuffer(outputImgBuffer, CL_TRUE, 0, outputImgVect.size(), &outputImgVect.data()[0], NULL, &outputProf);

	// Display comparison between input & output
	CImg<unsigned char> output_image(outputImgVect.data(), inputImgPtr.width(), inputImgPtr.height(), inputImgPtr.depth(), inputImgPtr.spectrum());
	CImgDisplay inputImgDisp(inputImgPtr, "[COLOUR] Input Image - IMP15591119");
	CImgDisplay outputImgDisp(output_image, "[COLOUR] Output Image - IMP15591119");

	cout << "[Part 4] Output Image Buffer Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 4] Look-Up Table Kernel Execution Time [ns]:" << lutProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lutProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "[Part 4] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(lutProf, ProfilingResolution::PROF_NS) << endl;


	while (!inputImgDisp.is_closed() && !outputImgDisp.is_closed() && !inputImgDisp.is_keyESC() && !outputImgDisp.is_keyESC()) {
		inputImgDisp.wait(1);
		inputImgDisp.wait(1);
	}

}

// Performs contrast adjustment for a greyscale image
void perform_greyscale_op(CImg<unsigned char> inputImgPtr, int platform_id, int device_id) {
	// Select platform and device to use to create a context from
	cl::Context context = GetContext(platform_id, device_id);

	// Display the selected device
	cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

	// Create a queue to which we will push commands for the device & enable profiling
	cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

	// Create a program to combine context and kernels
	cl::Program::Sources sources;
	AddSources(sources, "kernels/assign_kernels.cl");

	// Create a program to combine context and kernels
	cl::Program program(context, sources);

	// Attempt to build the OpenCL Program and catch any errors that occur during build
	try {
		program.build();
	}
	catch (const cl::Error& err) {
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		// Throw error and exit
		throw err;
		return;
	}

	// Get info about the device we're operating on
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];

	const int BIN_SIZE = 256; // Hard-coded bin size of 256
	cl::Event inputProf, outputProf; // Create generic CL Events for profiling

	/* PART 1 - Histogram Generation [GREYSCALE] */
	std::vector<int> histBin(BIN_SIZE); // Create a histogram to hold values 
	const size_t HIST_SIZE = histBin.size() * sizeof(int); // Hard-coded bin size

	// Create our initial buffers for usage in OpenCL Kernels
	cl::Buffer inputImgBuffer(context, CL_MEM_READ_ONLY, inputImgPtr.size()); // Create a read-only buffer with a size of our input image
	cl::Buffer histBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE);  // Create a read-write buffer with the size of our histogram bin (bin_size * size(int))
	
	// Write image input data to our device's memory via our image input buffer
	queue.enqueueWriteBuffer(inputImgBuffer, CL_TRUE, 0, inputImgPtr.size(), &inputImgPtr.data()[0], NULL, &inputProf);
	// Write histogram bin buffer filled with 0's to our device's memory
	queue.enqueueFillBuffer(histBuffer, 0, 0, HIST_SIZE);

	// Set up histogram kernel for device execution
	cl::Kernel kernelHist = cl::Kernel(program, "histogram"); // Load the histogram kernel defined in my_kernels
	kernelHist.setArg(0, inputImgBuffer);  // Pass in our image buffer as our input
	kernelHist.setArg(1, histBuffer);  // Pass in our histogram buffer as our output

	// Report stats for histogram kernel
	cout << "[Part 1] Maximum Work Group Size: ";
	cerr << kernelHist.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 1] Preferred Work Group Size: ";
	cerr << kernelHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event histogramProf;

	// Execute histogram kernel with attatched profiler
	queue.enqueueNDRangeKernel(kernelHist, cl::NullRange, cl::NDRange(inputImgPtr.size()), cl::NDRange(256), NULL, &histogramProf);
	// Write the histogram result from our device memory to our vector via the histogram buffer
	queue.enqueueReadBuffer(histBuffer, CL_TRUE, 0, histBin.size() * sizeof(int), &histBin[0], NULL, &outputProf);

	cout << "[Part 1] Image Buffer Memory Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 1] Histogram Buffer Memory Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 1] Histogram Kernel Execution Time [ns]:" << histogramProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogramProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "[Part 1] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(histogramProf, ProfilingResolution::PROF_NS) << endl;

	/* PART 2 - Cumulative Histogram Generation */

	// Create a new vector to store our cumulative bin values
	std::vector<int> cumBin(BIN_SIZE);

	// Create a new buffer to hold data about our cumulative histogram on our device
	cl::Buffer cumHistBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE);

	// Write histogram data to our device's memory via our histogram buffer
	queue.enqueueWriteBuffer(histBuffer, CL_TRUE, 0, HIST_SIZE, &histBin[0], NULL, &inputProf);
	// Write cumulative histogram bin buffer filled with 0's to our devices memory
	queue.enqueueFillBuffer(cumHistBuffer, 0, 0, HIST_SIZE);

	// Set up cumulative kernel for device execution
	cl::Kernel kernelCum = cl::Kernel(program, "scan_hs"); // Load the scan_hs kernel defined in assign_kernels
	kernelCum.setArg(0, histBuffer); // Pass in our histogram buffer as our input
	kernelCum.setArg(1, cumHistBuffer); // Pass in our cumulative histogram buffer as our output

	// Report stats for cumulative kernel
	cout << "[Part 2] Maximum Work Group Size: ";
	cerr << kernelCum.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 2] Preferred Work Group Size: ";
	cerr << kernelCum.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event cumulativeProf;

	// Execute the cumulative histogram kernel on the selected device
	queue.enqueueNDRangeKernel(kernelCum, cl::NullRange, cl::NDRange(histBin.size()), kernelCum.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device), NULL, &cumulativeProf);

	// Copy the result from device to host
	queue.enqueueReadBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &cumBin[0], NULL, &outputProf);

	cout << "[Part 2] Cumulative Histogram Buffer Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 2] Cumulative Histogram Buffer Output Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 2] Cumulative Kernel Execution Time [ns]:" << cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulativeProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "[Part 2] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(cumulativeProf, ProfilingResolution::PROF_NS) << endl;

	/* Part 3 - Cumulative Histogram Normalisation */

	// Create a new vector to store our cumulative bin values
	std::vector<int> normHistBin(cumBin.size());

	// Create a new buffer to hold data about our normalised cumulative histogram on our device
	cl::Buffer normHistBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE);
	cl::Buffer pixelCountBuffer(context, CL_MEM_READ_ONLY, sizeof(float)); // Create buffer to store normalisation calc

	float pixelCount = (float)255 / (float)(inputImgPtr.height() * inputImgPtr.width()); // Obtain pixel count of image

	// Write histogram data to our device's memory via our cumulative histogram buffer
	queue.enqueueWriteBuffer(cumHistBuffer, CL_TRUE, 0, HIST_SIZE, &cumBin[0], NULL, &inputProf);
	queue.enqueueWriteBuffer(pixelCountBuffer, CL_TRUE, 0, sizeof(float), &pixelCount);
	// Write normalised cumulative histogram bin buffer filled  with 0's to our devices memory
	queue.enqueueFillBuffer(normHistBuffer, 0, 0, HIST_SIZE);

	// Set up normalised cumulative kernel for device execution
	cl::Kernel kernelCumNormHist = cl::Kernel(program, "norm_bins"); // Load the norm_bins kernel defined in my_kernels
	kernelCumNormHist.setArg(0, cumHistBuffer); // Load in the cumulative histogram buffer
	kernelCumNormHist.setArg(1, normHistBuffer); // Pass in our normalised buffer filled with 0's
	kernelCumNormHist.setArg(2, pixelCountBuffer); // Pass in our calculated normalisation value TODO: ASK

	// Report stats for normalisation kernel
	cout << "[Part 3] Maximum Work Group Size: ";
	cerr << kernelCumNormHist.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 3] Preferred Work Group Size: ";
	cerr << kernelCumNormHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event normalisedProf;

	// Execute the cumulative histogram kernel on the selected device
	queue.enqueueNDRangeKernel(kernelCumNormHist, cl::NullRange, cl::NDRange(cumBin.size()), kernelCumNormHist.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device), NULL, &normalisedProf);

	// Copy the result from device to host
	queue.enqueueReadBuffer(normHistBuffer, CL_TRUE, 0, HIST_SIZE, &normHistBin[0], NULL, &outputProf);

	cout << "[Part 3] Normalised Histogram Buffer Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 3] Normalised Histogram Buffer Output Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 3] Normalised Kernel Execution Time [ns]:" << normalisedProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - normalisedProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "[Part 3] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(normalisedProf, ProfilingResolution::PROF_NS) << endl;

	/* Part 4 - Image from LUT */

	// Create an output buffer to store values copied from device once computation is complete
	vector<unsigned char> outputImgVect(inputImgPtr.size());
	// Create a new buffer to hold data about our output image
	cl::Buffer outputImgBuffer(context, CL_MEM_READ_WRITE, inputImgPtr.size()); //should be the same as input image

	// Write normalised cumulative histogram data to our predefined buffer
	queue.enqueueWriteBuffer(normHistBuffer, CL_TRUE, 0, HIST_SIZE, &normHistBin[0], NULL, &inputProf);

	cl::Kernel kernelLut = cl::Kernel(program, "lut"); // Load the LUT kernel defined in my_kernels
	kernelLut.setArg(0, inputImgBuffer); // Load in our normalised histogram buffer bin
	kernelLut.setArg(1, outputImgBuffer); // Load in our input image in buffer form
	kernelLut.setArg(2, normHistBuffer); // Load in our output image buffer for writing to

	// Report stats for normalisation kernel
	cout << "[Part 4] Maximum Work Group Size: ";
	cerr << kernelLut.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // Get device info
	cout << "[Part 4] Preferred Work Group Size: ";
	cerr << kernelLut.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // Get device info

	// Event for tracking kernel execution time
	cl::Event lutProf;

	// Execute the look-up table histogram kernel on the selected device
	queue.enqueueNDRangeKernel(kernelLut, cl::NullRange, cl::NDRange(inputImgPtr.size()), cl::NDRange(256), NULL, &lutProf);

	//4.3 Copy the result from device to host
	queue.enqueueReadBuffer(outputImgBuffer, CL_TRUE, 0, outputImgVect.size(), &outputImgVect.data()[0], NULL, &outputProf);

	CImg<unsigned char> output_image(outputImgVect.data(), inputImgPtr.width(), inputImgPtr.height(), inputImgPtr.depth(), inputImgPtr.spectrum());
	CImgDisplay inputImgDisp(inputImgPtr, "[GREY] Input Image - IMP15591119");
	CImgDisplay outputImgDisp(output_image, "[GREY] Output Image - IMP15591119");

	cout << "[Part 4] Input Image Buffer Write Time [ns]: " << inputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 4] Output Image Buffer Write Time [ns]: " << outputProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - outputProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "[Part 4] Look-Up Table Kernel Execution Time [ns]:" << lutProf.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lutProf.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "[Part 4] Full Profiling Info (kernel) [ns]: " << GetFullProfilingInfo(lutProf, ProfilingResolution::PROF_NS) << endl;

	while (!inputImgDisp.is_closed() && !outputImgDisp.is_closed() && !inputImgDisp.is_keyESC() && !outputImgDisp.is_keyESC()) {
		inputImgDisp.wait(1);
		inputImgDisp.wait(1);
	}
}