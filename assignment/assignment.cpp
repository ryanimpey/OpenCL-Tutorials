#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

// Returns console information about different flags that can be passed to the function
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	// Define platform and device ID's as defaults incase no arguements are passed through
	int platform_id = 0;
	int device_id = 0;

	// Load in our initial reference file
	string inputImgFilename = "test.pgm";

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

		// Select platform and device to use to create a context from
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		// Create a program to combine context and kernels
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");

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
			throw err;
		}

		/* Part 1 - Histogram Generation */

		// Create a host vector (Histogram Bin) to hold our output values
		std::vector<int> histBin(256);
		size_t histBinSize = histBin.size() * sizeof(int);

		// Create our buffers for usage in OpenCL Kernels
		cl::Buffer inputImgBuffer(context, CL_MEM_READ_ONLY, inputImgPtr.size()); // Create a read-only buffer with a size of our input image
		cl::Buffer histBuffer(context, CL_MEM_READ_WRITE, histBinSize);  // Create a read-write buffer with the size of our histogram bin (bin_size * size(int))
//		cl::Buffer image_output_buffer(context, CL_MEM_READ_ONLY, inputImgPtr.size()); // Not in use yet!
//		cl::Buffer scan_histogram_buffer(context, CL_MEM_READ_WRITE, histBinSize); // Not in use yet!

		std::cout << "Bin Size: " << histBin.size() << std::endl;
		std::cout << "Bin Size in Bytes: " << histBinSize << std::endl;

		// Write image input data to our device's memory via our image input buffer
		queue.enqueueWriteBuffer(inputImgBuffer, CL_TRUE, 0, inputImgPtr.size(), &inputImgPtr.data()[0]);
		// Write histogram bin buffer filled with 0's to our device's memory
		queue.enqueueFillBuffer(histBuffer, 0, 0, histBinSize);

		// Set up histogram kernel for device execution
		cl::Kernel kernel_histogram = cl::Kernel(program, "histogram"); // Load the histogram kernel defined in my_kernels
		kernel_histogram.setArg(0, inputImgBuffer);  // Pass in our image buffer as our input
		kernel_histogram.setArg(1, histBuffer);  // Pass in our histogram buffer as our output

		// Execute our histogram kernel on the selected device
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(inputImgPtr.size()), cl::NullRange);

		// Write the histogram result from our device memory to our vector via the histogram buffer
		queue.enqueueReadBuffer(histBuffer, CL_TRUE, 0, histBinSize, &histBin[0]);

		std::cout << "Histogram Bin: " << histBin << "\n\n" << std::endl;

		// Create an output buffer to store values copied from device once computation is complete
		////vector<unsigned char> output_buffer(inputImgPtr.size());

		// Copy result to the buffer just defined
		////queue.enqueueReadBuffer(image_output_buffer, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]); // Not needed yet
		

		/* PART TWO - CUMULATIVE HISTOGRAM */

		

		//std::vector<int> bin_test(256, 1);

		//std::cout << "Bin Test:" << bin_test << "\n" << std::endl;


		//// Fill histogram bin buffer with 0's on device memory
		//queue.enqueueFillBuffer(scan_histogram_buffer, 0, 0, histBinSize);

		//queue.enqueueWriteBuffer(histBuffer, CL_TRUE, 0, bin_test.size() * sizeof(int), &bin_test[0]);

		//// Redefine the kernel as our second step
		//cl::Kernel kernel_scan = cl::Kernel(program, "scan_add");
		//kernel_scan.setArg(0, histBuffer);
		//kernel_scan.setArg(1, scan_histogram_buffer);
		//kernel_scan.setArg(2, cl::Local(bin_test.size() * sizeof(int)));//local memory size
		//kernel_scan.setArg(3, cl::Local(bin_test.size() * sizeof(int)));//local memory size

		//// Enqueues a command to execute a kernel on a device
		//queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(histBinSize), cl::NDRange(256));

		//queue.enqueueReadBuffer(scan_histogram_buffer, CL_TRUE, 0, bin_test.size(), &bin[0]);

		//std::cout << "\nCumulative Bin: " << bin << std::endl;

		// Create a new CImg and window to display the results
		////CImg<unsigned char>image_output(output_buffer.data(), inputImgPtr.width(), inputImgPtr.height(), inputImgPtr.depth(), inputImgPtr.spectrum());
		////CImgDisplay disp_output(image_output, "output");

		// Requires both the input and output image to be closed before the application is terminated
		/*
		while (!disp_input.is_closed() && !disp_output.is_closed() && !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
		*/
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
