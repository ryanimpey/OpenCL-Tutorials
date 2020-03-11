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
	string image_filename = "test.pgm";

	// Handle command line arguements
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	// Hides CImg library messages/exceptions from the output
	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		// Returns a pointer to a image location from its filename
		CImg<unsigned char> image_input(image_filename.c_str());

		// Select platform and device to use to create a context from
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");

		// Create a program to combine context and kernels
		cl::Program program(context, sources);

		// Attempt to build the program
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		// Create a host vector (Histogram Bin) to hold our output values
		std::vector<int> bin(256);
		size_t bin_size = bin.size() * sizeof(int);

		// Create two buffers, one for the input image and one for the output
		cl::Buffer image_input_buffer(context, CL_MEM_READ_ONLY, image_input.size());
		////cl::Buffer image_output_buffer(context, CL_MEM_READ_ONLY, image_input.size());

		// Create a third bin to hold our histogram values
		cl::Buffer image_histogram_buffer(context, CL_MEM_READ_WRITE, bin_size);
		cl::Buffer scan_histogram_buffer(context, CL_MEM_READ_WRITE, bin_size);

		std::cout << "Bin Size: " << bin.size() << std::endl;
		std::cout << "Bin Size in Bytes: " << bin_size << std::endl;

		// Copy input buffer to device memory
		queue.enqueueWriteBuffer(image_input_buffer, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		// Fill histogram bin buffer with 0's on device memory
		queue.enqueueFillBuffer(image_histogram_buffer, 0, 0, bin_size);

		// Set up kernel for device execution and pass in buffers as arguements
		cl::Kernel kernel_histogram = cl::Kernel(program, "histogram");
		kernel_histogram.setArg(0, image_input_buffer);
		kernel_histogram.setArg(1, image_histogram_buffer);

		// Enqueues a command to execute a kernel on a device
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);


		// Create an output buffer to store values copied from device once computation is complete
		////vector<unsigned char> output_buffer(image_input.size());

		// Copy result to the buffer just defined
		////queue.enqueueReadBuffer(image_output_buffer, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]); // Not needed yet
		// Copy result back to our original 'bin' vector
		queue.enqueueReadBuffer(image_histogram_buffer, CL_TRUE, 0, bin_size, &bin[0]);

		// ---

		std::cout << "Histogram Bin: " << bin << "\n\n" << std::endl;

		std::vector<int> bin_test(256, 1);

		std::cout << "Bin Test:" << bin_test << "\n" << std::endl;

		// Fill histogram bin buffer with 0's on device memory
		queue.enqueueFillBuffer(scan_histogram_buffer, 0, 0, bin_size);
		queue.enqueueFillBuffer(image_histogram_buffer, 0, 0, bin_size);

		queue.enqueueWriteBuffer(image_histogram_buffer, CL_TRUE, 0, bin_test.size(), &bin_test[0]);

		// Redefine the kernel as our second step
		cl::Kernel kernel_scan = cl::Kernel(program, "scan_hs");
		kernel_scan.setArg(0, image_histogram_buffer);
		kernel_scan.setArg(1, scan_histogram_buffer);

		// Enqueues a command to execute a kernel on a device
		queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(bin_size), cl::NDRange(256));

		queue.enqueueReadBuffer(scan_histogram_buffer, CL_TRUE, 0, bin_test.size(), &bin[0]);

		std::cout << "\nCumulative Bin: " << bin << std::endl;

		// Create a new CImg and window to display the results
		////CImg<unsigned char>image_output(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
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
