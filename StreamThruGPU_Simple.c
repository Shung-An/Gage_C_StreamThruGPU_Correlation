/////////////////////////////////////////////////////////////////////////
//
// GageStreamThruGPU_Simple
//
// See the official site at www.gage-applied.com for documentation and
// the latest news.
//
/////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 1994-2019 by Gage Applied Technologies.
// All rights reserved.
//
// This code is free for personal and commercial use, providing this
// notice remains intact in the source files and all eventual changes are
// clearly marked with comments.
//
// No warranty of any kind, expressed or implied, is included with this
// software; use at your own risk, responsibility for damages (if any) to
// anyone resulting from the use of this software rests entirely with the
// user.
//
/////////////////////////////////////////////////////////////////////////
//
// GageStreamThruGPU_Simple demonstrates the Streamming to Disk capabilities of 
// CompuScope data acquisition system.  The samples program will setup a Compuscope 
// system to Data Streaming mode. All data captured will be saved
// into hard disk. Additionally the program demonstrates how to integrate
// CUDA functionality into the program by using a very simple CUDA kernel.  The
// included kernel is easy to relace with a more complex version.
//
///////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>
#include <stdio.h>
#include "CsPrototypes.h"
#include "C:\Program Files (x86)\Gage\CompuScope\CompuScope C SDK\C Common\CsAppSupport.h"
#include "C:\Program Files (x86)\Gage\CompuScope\CompuScope C SDK\C Common\CsSdkMisc.h"
#include "C:\Program Files (x86)\Gage\CompuScope\CompuScope C SDK\C Common\CsTchar.h"
#include "CsExpert.h"
#include <time.h>
#include <io.h>
#include <pthread.h>


#define	MAX_CARDS_COUNT			10				// Max number of cards supported in a M/S Compuscope system 
#define	SEGMENT_TAIL_ADJUST	64					// number of bytes at end of data which holds the timestamp values
#define OUT_FILE	"Data"						// name of the output file 
#define LOOP_COUNT	1000
#define TRANSFER_TIMEOUT	10000				
#define STREAM_BUFFERSZIZE	0x200000
#define STM_SECTION _T("StmConfig")				// section name in ini file


// User configuration variables
typedef struct
{
	uInt32		u32BufferSizeBytes;
	uInt32		u32TransferTimeout;
	uInt32		u32DelayStartTransfer;
	uInt32		u32TimeCounter;
	TCHAR		strResultFile[MAX_PATH];
	BOOL		bSaveToFile;			// Save data to file or not
	BOOL		bFileFlagNoBuffering;	// Should be 1 for better disk performance
	BOOL		bErrorHandling;			// How to handle the FIFO full error
	BOOL		bCascadeResult;			// Do cascade the result without initilizing a new file
	CsDataPackMode		DataPackCfg;
}CSSTMCONFIG, *PCSSTMCONFIG;


#define GPU_SECTION _T("GpuConfig")		/* section name in ini file */
#define RESULTS_FILE _T("Result")
#define	MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

#define DEFAULT_SKIP_FACTOR 1
#define DEFAULT_DO_ANALYSIS 1
#define DEFAULT_USE_GPU 1

typedef struct
{
	int32		i32GpuBlocks;
	int32		i32GpuThreads;
	uInt32		u32SkipFactor;
	TCHAR		strResultFile[MAX_PATH];
	BOOL		bDoAnalysis;			/* Turn on or off data analysis */
	BOOL		bUseGpu;				/* Turn on or off GPU usage */
}CSGPUCONFIG, *PCSGPUCONFIG;



int32 InitializeStream(CSHANDLE hSystem);
uInt32 CalculateTriggerCountFromConfig(CSSYSTEMINFO* pCsSysInfo, const LPCTSTR szIniFile);

BOOL isChannelValid(uInt32 u32ChannelIndex, uInt32 u32mode, uInt16 u16cardIndex, CSSYSTEMINFO* pCsSysInfo);
uInt32 GetSectorSize();

void  UpdateProgress( uInt32 u32Elapsed, LONGLONG llTotaBytes );
int32 LoadStmConfiguration(LPCTSTR szIniFile, PCSSTMCONFIG pConfig);
DWORD WINAPI CardStreamThread( void *CardIndex );
BOOL Prepare_Cleanup();

#ifdef __cplusplus
extern "C" cudaError_t GPU_Equation_PlusOne(void* a, unsigned long skip, unsigned long sample_size, __int64 size, int blocks, int threads);
extern "C" int CPU_Equation_PlusOne(void* buffer, unsigned long sample_size, __int64 start, __int64 length);
#else
extern cudaError_t GPU_Equation_PlusOne(void* a, unsigned long skip, unsigned long sample_size, __int64 size, int blocks, int threads);
extern int CPU_Equation_PlusOne(void* buffer, unsigned long sample_size, __int64 start, __int64 length);
#endif



int32 LoadGpuConfiguration(LPCTSTR szIniFile, PCSGPUCONFIG pConfig);
cudaError_t InitializeCudaDevice(int32 nDevice, int32* i32MaxBlocks, int32* i32MaxThreads, BOOL* bPinGenericMemory);
void DisplayResults(int stream,
					int gpu,
					uInt32 u32Mode,
					uInt32 u32SegmentCount,
					int64 i64TransferLength,
					uInt32 u32SampleSize,
					uInt32 u32SkipFactor,
					double time,
					char* filename);

void VerifyData(void* buffer, int64 size, unsigned int sample_size);


// Global variables shared between threads
HANDLE						g_hThread[MAX_CARDS_COUNT] = {0};
LONGLONG					g_llCardTotalData[MAX_CARDS_COUNT] = {0};
LONGLONG					g_llTotalSamplesConfig = 0;
HANDLE						g_hStreamStarted = NULL;
HANDLE						g_hStreamAbort = NULL;
HANDLE						g_hStreamError = NULL;
HANDLE						g_hThreadReadyForStream = NULL;
CSHANDLE					g_hSystem = 0;
CSSYSTEMINFO				g_CsSysInfo = {0};
CSACQUISITIONCONFIG			g_CsAcqCfg = {0};
CSSTMCONFIG					g_StreamConfig = {0};
CSGPUCONFIG					g_GpuConfig = {0};
CS_STRUCT_DATAFORMAT_INFO	g_DataFormatInfo = {0};
double						diff_time[MAX_CARDS_COUNT] = { 0. };

int _tmain()
{
	int32						i32Status = CS_SUCCESS;
	uInt32						u32Mode;
	CSSYSTEMINFO				CsSysInfo = {0};
	LPCTSTR						szIniFile = _T("StreamThruGPU.ini");
	BOOL						bDone = FALSE;
	long long					llSystemTotalData = 0;
	uInt16						n;
	uInt16						i;
	DWORD						dwWaitStatus;
	DWORD						dwThreadId;
	uInt32						u32TickStart = 0;
	uInt32						u32TickNow = 0;
	double						dTotalData = 0.0;

	BOOL						bStreamPackedSupported = FALSE;

	int32						i32MaxThreadsPerBlock;
	int32						i32MaxBlocks;
	BOOL						bPinGenericMemory = TRUE;
	int32						i32CudaDevice = 0;
	cudaError_t					cudaStatus = cudaSuccess;
	int64						i64TickFrequency = 0;

	// Initializes the CompuScope boards found in the system. If the
	// system is not found a message with the error code will appear.
	// Otherwise i32Status will contain the number of systems found.
	i32Status = CsInitialize();

	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		return (-1);
	}

	// Get the system. This sample program only supports one system. If
	// 2 systems or more are found, the first system that is found
	// will be the system that will be used. g_hSystem will hold a unique
	// system identifier that is used when referencing the system.
	i32Status = CsGetSystem(&g_hSystem, 0, 0, 0, 0);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		return (-1);
	}

	// Get System information. The u32Size field must be filled in
	// prior to calling CsGetSystemInfo
	CsSysInfo.u32Size = sizeof(CSSYSTEMINFO);
	i32Status = CsGetSystemInfo(g_hSystem, &CsSysInfo);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}
	// A copy used in the threads to build headers
	g_CsSysInfo = CsSysInfo;

	// Display the system name from the driver
	_ftprintf(stdout, _T("\nBoard Name: %s"), CsSysInfo.strBoardName);

	//	We are analysing the ini file to find the number of triggers
	i32Status = CsAs_ConfigureSystem(g_hSystem, (int)CsSysInfo.u32ChannelCount, 
		(int)CalculateTriggerCountFromConfig(&CsSysInfo,(LPCTSTR)szIniFile), 
		(LPCTSTR)szIniFile, &u32Mode);

	if (CS_FAILED(i32Status))
	{
		if (CS_INVALID_FILENAME == i32Status)
		{
			// Display message but continue on using defaults.
			_ftprintf(stdout, _T("\nCannot find %s - using default parameters."), szIniFile);
		}
		else
		{	

			// Otherwise the call failed.  If the call did fail we should free the CompuScope
			// system so it's available for another application
			DisplayErrorString(i32Status);
			CsFreeSystem(g_hSystem);
			return(-1);
		}
	}
	
	// If the return value is greater than  1, then either the application, 
	// acquisition, some of the Channel and / or some of the Trigger sections
	// were missing from the ini file and the default parameters were used. 
	if (CS_USING_DEFAULT_ACQ_DATA & i32Status)
		_ftprintf(stdout, _T("\nNo ini entry for acquisition. Using defaults."));

	if (CS_USING_DEFAULT_CHANNEL_DATA & i32Status)
		_ftprintf(stdout, _T("\nNo ini entry for one or more Channels. Using defaults for missing items."));

	if (CS_USING_DEFAULT_TRIGGER_DATA & i32Status)
		_ftprintf(stdout, _T("\nNo ini entry for one or more Triggers. Using defaults for missing items."));


	// Load application specific information from the ini file
	i32Status = LoadStmConfiguration(szIniFile, &g_StreamConfig);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}
	if (0 != g_StreamConfig.DataPackCfg)
	{
		_ftprintf(stdout, _T("\nThis program does not support packed data. Resetting to unpacked mode."));
		g_StreamConfig.DataPackCfg = 0;
	}
	if (CS_USING_DEFAULTS == i32Status)
	{
		_ftprintf(stdout, _T("\nNo ini entry for Stm configuration. Using defaults."));
	}

	i32Status = LoadGpuConfiguration(szIniFile, &g_GpuConfig);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}
	if (CS_USING_DEFAULTS == i32Status)
	{
		_ftprintf(stdout, _T("\nNo ini entry for Gpu configuration. Using defaults."));
	}

	// if we're not doing analysis then we're not using the GPU
	if (!g_GpuConfig.bDoAnalysis)
	{
		g_GpuConfig.bUseGpu = FALSE;
	}

	// bUseGpu being true implies that we're doing analysis
	if (g_GpuConfig.bUseGpu)
	{
		// Initialize cuda device.  If i32MaxBlocks or i32MaxThreadsPerBlock were -1 we set them to the maximum
		cudaStatus = InitializeCudaDevice(i32CudaDevice, &i32MaxBlocks, &i32MaxThreadsPerBlock, &bPinGenericMemory);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "\ncuda error %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			return (-1);
		}
		// our streaming uses a buffer allocated by our kernel driver, so we need to be able to map it
		// onto the GPU using cudaHostRegister and cudaGetDevicePointer
		if (!bPinGenericMemory)
		{
			fprintf(stderr, "\nPinned generic memory is needed for streaming but is not available.\nExiting program\n");
			CsFreeSystem(g_hSystem);
			return (-1);
		}

		if (g_GpuConfig.i32GpuBlocks > i32MaxBlocks || -1 == g_GpuConfig.i32GpuBlocks)
		{
			fprintf(stderr, "\nBlock count is too big, changed from %d blocks to %d blocks\n", g_GpuConfig.i32GpuBlocks, i32MaxBlocks);
			g_GpuConfig.i32GpuBlocks = i32MaxBlocks;
		}
		if (-1 == g_GpuConfig.i32GpuBlocks) // -1 means use the maximum
		{
			g_GpuConfig.i32GpuBlocks = i32MaxBlocks;
		}
		if (g_GpuConfig.i32GpuThreads > i32MaxThreadsPerBlock)
		{
			fprintf(stderr, "\nThreads per block is too big, changed from %d threads to %d threads\n", g_GpuConfig.i32GpuThreads, i32MaxThreadsPerBlock);
			g_GpuConfig.i32GpuThreads = i32MaxThreadsPerBlock;
		}
		if (-1 == g_GpuConfig.i32GpuThreads) // -1 means use the maximum
		{
			g_GpuConfig.i32GpuThreads = i32MaxThreadsPerBlock;
		}
		if (0 == g_GpuConfig.i32GpuThreads)
		{
			_ftprintf(stderr, "\nGPU thread count cannot be 0, changed to 1\n");
			g_GpuConfig.i32GpuThreads = 1;
		}
	}

	// Streaming Configuration.
	// Validate if the board supports hardware streaming. If  it is not supported, 
	// we'll exit gracefully.
	i32Status = InitializeStream(g_hSystem);
	if (CS_FAILED(i32Status))
	{
		// Error string was displayed in InitializeStream
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	 // Prepare streaming by deleting all existing data file that have the same file name
	 if ( 0 == Prepare_Cleanup() )
	 {
		 CsFreeSystem(g_hSystem);
		return (-1);
	}


	// Create events for stream data acquisition
	g_hStreamStarted	= CreateEvent( NULL, TRUE, FALSE, NULL );
	g_hStreamAbort		= CreateEvent( NULL, TRUE, FALSE, NULL );
	g_hStreamError		= CreateEvent( NULL, TRUE, FALSE, NULL );
	g_hThreadReadyForStream	= CreateEvent( NULL, FALSE, FALSE, NULL );
	if ( NULL == g_hStreamStarted || NULL == g_hStreamAbort || NULL == g_hStreamError || NULL == g_hThreadReadyForStream )
	{
		_ftprintf (stderr, _T("\nUnable to create events for synchronization.\n"));
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	// Commit the values to the driver.  This is where the values get sent to the
	// hardware.  Any invalid parameters will be caught here and an error returned.
	i32Status = CsDo(g_hSystem, ACTION_COMMIT);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	// After ACTION_COMMIT, the sample size may change.
	// Get user's acquisition data to use for various parameters for transfer
	g_CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG);
	i32Status = CsGet(g_hSystem, CS_ACQUISITION, CS_CURRENT_CONFIGURATION, &g_CsAcqCfg);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	// Get the total amount of data we expect to receive.
	// We can get this value from driver or calculate the following formula
	// g_llTotalSamplesConfig = (g_CsAcqCfg.i64SegmentSize + SegmentTail_Size) * (g_CsAcqCfg.u32Mode&CS_MASKED_MODE) * g_CsAcqCfg.u32SegmentCount;
	
	i32Status = CsGet( g_hSystem, 0, CS_STREAM_TOTALDATA_SIZE_BYTES, &g_llTotalSamplesConfig );
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	// Convert to number of samples
	if ( -1 != g_llTotalSamplesConfig )
		g_llTotalSamplesConfig /= g_CsAcqCfg.u32SampleSize;

	//  Create threads for Stream. In M/S system, we have to create one thread per card
	for (n = 1, i = 0; n <= CsSysInfo.u32BoardCount; n++, i++ )
	{
		g_hThread[i] = (HANDLE) CreateThread( NULL, 0, CardStreamThread, &n, 0, &dwThreadId );
		if ( (HANDLE)(INT_PTR) -1 == g_hThread[i] )
		{
			// Fail to create the streaming thread for the n card.
			// Set the event g_hStreamAbort to terminate all threads
			SetEvent( g_hStreamAbort ); 
			_ftprintf (stderr, _T("\nUnable to create thread for card %d."), n);
			CsFreeSystem(g_hSystem);
			return (-1);
		}
		else
		{
			// Wait for the event g_hThreadReadyForStream to make sure that the thread was successfully created and are ready for stream
			if (WAIT_TIMEOUT == WaitForSingleObject( g_hThreadReadyForStream,  10000 ))
			{
				// Something is wrong. It is not suppose to take that long
				// Set the event g_hStreamAbort to terminate all threads
				_ftprintf (stderr, _T("\nThread initialization error on card %d."), n);
				SetEvent( g_hStreamAbort ); 
				CsFreeSystem(g_hSystem);
				return (-1);
			}
		}
	}

	// Start the streaming data acquisition
	printf ("\nStart streaming. Press ESC to abort\n\n");
	i32Status = CsDo(g_hSystem, ACTION_START);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		CsFreeSystem(g_hSystem);
		return (-1);
	}

	u32TickStart = u32TickNow = GetTickCount();

	// Set the event g_hStreamStarted so that the other threads can start to transfer data
	Sleep(g_StreamConfig.u32DelayStartTransfer);		// Only for debug
	SetEvent( g_hStreamStarted );


	// Loop until either we've done the number of segments we want, or
	// the ESC key was pressed to abort.
	while( !bDone  )
	{
		// Are we being asked to quit? 
		if (_kbhit())
		{
			switch (toupper(_getch()))
			{
			case 27:			// ESC key -> abort
				SetEvent(g_hStreamAbort);
				bDone = TRUE;
				break;
			case 'F':			// F key -> force trigger
				i32Status = CsDo(g_hSystem, ACTION_FORCE);
				if (CS_FAILED(i32Status))
					DisplayErrorString(i32Status);
			default:
				MessageBeep(MB_ICONHAND);
				break;
			}
		}
		// Quit if elapsed time greater than our setting. 
		if (u32TickNow - u32TickStart >= g_StreamConfig.u32TimeCounter)
		{
			SetEvent(g_hStreamAbort);
			bDone = TRUE;
		}

		dwWaitStatus = WaitForMultipleObjects( CsSysInfo.u32BoardCount, g_hThread, TRUE, 1000 );
		if ( WAIT_OBJECT_0 == dwWaitStatus )
		{
			// All Streaming threads have terminated
			bDone = TRUE;
		}

		u32TickNow = GetTickCount();

		// Calcaulate the sum of all data received so far
		llSystemTotalData = 0;
		for (i = 0; i < CsSysInfo.u32BoardCount; i++ )
		{
			llSystemTotalData += g_llCardTotalData[i];
		}

		UpdateProgress(u32TickNow-u32TickStart, llSystemTotalData * g_CsSysInfo.u32SampleSize);
	}

	//	Abort the current acquisition 
	CsDo(g_hSystem, ACTION_ABORT);

	// Free the CompuScope system and any resources it's been using
	i32Status = CsFreeSystem(g_hSystem);

	if (g_GpuConfig.bUseGpu)
	{
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		// We're doing it here because if we don't we can't access the buffer in unpinned mode.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}

	// Check some events to see if there was any errors
	if (WAIT_OBJECT_0 == WaitForSingleObject( g_hStreamError, 0 ) )
	{
		_ftprintf (stdout, _T("\nStream aborted on error.\n"));
	}
	else if (WAIT_OBJECT_0 == WaitForSingleObject( g_hStreamAbort, 0 ))
	{
		_ftprintf (stdout, _T("\nStream aborted by user.\n"));
	}
	else
	{
		_ftprintf (stdout, _T("\n\nStream has finished %u segments.\n"), g_CsAcqCfg.u32SegmentCount);
	}

	dTotalData = 1.0*llSystemTotalData/1000000.0;		// Million BYTES or million WORDs depending on CompuScope cards

	if ( bStreamPackedSupported )
		printf ("\nTotal data in '%d-bit' samples: %0.2f MS\n", g_DataFormatInfo.u32SampleSize_Bits, dTotalData*g_CsSysInfo.u32SampleSize*8/g_DataFormatInfo.u32SampleSize_Bits );
	else
		printf ("\nTotal data in '%d-bit' samples: %0.2f MS\n", 8*g_CsAcqCfg.u32SampleSize, dTotalData );

	if (g_GpuConfig.bDoAnalysis)
	{
		long long llSystemTotalData = 0;
		double dSystemTotalTime = 0.;

		for (i = 0; i < CsSysInfo.u32BoardCount; i++)
		{
			llSystemTotalData += g_llCardTotalData[i];
			dSystemTotalTime += diff_time[i];
		}

 		DisplayResults(1,
				       g_GpuConfig.bUseGpu,
					   g_CsAcqCfg.u32Mode & CS_MASKED_MODE, // mask of the constant that loaded the expert firmware
					   g_CsAcqCfg.u32SegmentCount,
					   llSystemTotalData,
					   g_CsAcqCfg.u32SampleSize,
					   g_GpuConfig.u32SkipFactor,
					   dSystemTotalTime,
				       g_GpuConfig.strResultFile);
	}

	return 0;
}

/***************************************************************************************************
****************************************************************************************************/

uInt32 CalculateTriggerCountFromConfig(CSSYSTEMINFO* pCsSysInfo, const LPCTSTR szIniFile)
{
	TCHAR	szFilePath[MAX_PATH];
	TCHAR	szTrigger[100];
	TCHAR	szString[100];
	uInt32	i = 0;

	GetFullPathName(szIniFile, MAX_PATH, szFilePath, NULL);

	for( ; i < pCsSysInfo->u32TriggerMachineCount; ++i)
	{
		_stprintf(szTrigger, _T("Trigger%i"), i+1);
		
		if (0 == GetPrivateProfileSection(szTrigger, szString, 100, szFilePath))
			break;
	}

	return i;
}

/***************************************************************************************************
****************************************************************************************************/

BOOL isChannelValid(uInt32 u32ChannelIndex, uInt32 u32mode, uInt16 u16cardIndex, CSSYSTEMINFO* pCsSysInfo)
{
	uInt32 mode = u32mode & CS_MASKED_MODE;
	uInt32 channelsPerCard = pCsSysInfo->u32ChannelCount / pCsSysInfo->u32BoardCount;
	uInt32 min = ((u16cardIndex-1) * channelsPerCard) + 1;
	uInt32 max = (u16cardIndex * channelsPerCard);

	if((u32ChannelIndex-1) % (pCsSysInfo->u32ChannelCount / mode) != 0)
		return FALSE;

	return (u32ChannelIndex >= min && u32ChannelIndex <= max);
}

/***************************************************************************************************
****************************************************************************************************/

uInt32 GetSectorSize()
{
	uInt32 size = 0;
	if( !GetDiskFreeSpace(NULL, NULL, &size, NULL, NULL))
		return 0;
	return size;
}

/***************************************************************************************************
****************************************************************************************************/

int32 InitializeStream(CSHANDLE hSystem)
{
	int32	i32Status = CS_SUCCESS;
	int64	i64ExtendedOptions = 0;	
	char	szExpert[64];
	uInt32	u32ExpertOption = 0;
	CSACQUISITIONCONFIG CsAcqCfg = {0};


	u32ExpertOption = CS_BBOPTIONS_STREAM; 
	strcpy_s(szExpert, sizeof(szExpert), "Stream" );

	CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG);

	// Get user's acquisition Data
	i32Status = CsGet(hSystem, CS_ACQUISITION, CS_CURRENT_CONFIGURATION, &CsAcqCfg);
 	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		return (i32Status);
	}

	// Check if selected system supports Expert Stream
	// And set the correct image to be used.
	CsGet(hSystem, CS_PARAMS, CS_EXTENDED_BOARD_OPTIONS, &i64ExtendedOptions);

	if (i64ExtendedOptions & u32ExpertOption)
	{
		_ftprintf(stdout, _T("\nSelecting Expert %s from image 1."), szExpert);
		CsAcqCfg.u32Mode |= CS_MODE_USER1;
	}
	else if ((i64ExtendedOptions >> 32) & u32ExpertOption)
	{
		_ftprintf(stdout, _T("\nSelecting Expert %s from image 2."), szExpert);
		CsAcqCfg.u32Mode |= CS_MODE_USER2;
	}
	else
	{
		_ftprintf(stdout, _T("\nCurrent system does not support Expert %s."), szExpert);
		_ftprintf(stdout, _T("\nApplication terminated."));
		return CS_MISC_ERROR;
	}
	
	// Sets the Acquisition values down the driver, without any validation, 
	// for the Commit step which will validate system configuration.
	i32Status = CsSet(hSystem, CS_ACQUISITION, &CsAcqCfg);
	if (CS_FAILED(i32Status))
	{
		DisplayErrorString(i32Status);
		return CS_MISC_ERROR;
	}

	return CS_SUCCESS; // Success
}

/***************************************************************************************************
****************************************************************************************************/

void UpdateProgress( uInt32 u32Elapsed, LONGLONG llTotaBytes )
{
	uInt32	h = 0;
	uInt32	m = 0;
	uInt32	s = 0;
	double	dRate;
	double	dTotal;

	if ( u32Elapsed > 0 )
	{
		dRate = (llTotaBytes / 1000000.0) / (u32Elapsed / 1000.0);

		if ( u32Elapsed >= 1000 )
		{
			if ((s = u32Elapsed / 1000) >= 60)	// Seconds
			{
				if ((m = s / 60) >= 60)			// Minutes
				{
					if ((h = m / 60) > 0)		// Hours
						m %= 60;
				}
				s %= 60;
			}
		}
		dTotal = 1.0*llTotaBytes/1000000.0;		// Mega samples
		//printf ("\rTotal: %0.2f MB, Rate: %6.2f MB/s, Elapsed time: %u:%02u:%02u  ", dTotal, dRate, h, m, s);
	}
}

/***************************************************************************************************
****************************************************************************************************/

int32 LoadStmConfiguration(LPCTSTR szIniFile, PCSSTMCONFIG pConfig)
{
	TCHAR	szDefault[MAX_PATH];
	TCHAR	szString[MAX_PATH];
	TCHAR	szFilePath[MAX_PATH];
	int		nDummy;
	CSSTMCONFIG CsStmCfg = {0};	

	// Set defaults in case we can't read the ini file
	CsStmCfg.u32BufferSizeBytes = STREAM_BUFFERSZIZE;
	CsStmCfg.u32TransferTimeout = TRANSFER_TIMEOUT;
	strcpy(CsStmCfg.strResultFile, _T(OUT_FILE));
	
	if (NULL == pConfig)
	{
		return (CS_INVALID_PARAMETER);
	}

	GetFullPathName(szIniFile, MAX_PATH, szFilePath, NULL);

	if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(szFilePath))
	{
		*pConfig = CsStmCfg;
		return (CS_USING_DEFAULTS);
	}

	if (0 == GetPrivateProfileSection(STM_SECTION, szString, 100, szFilePath))
	{
		*pConfig = CsStmCfg;
		return (CS_USING_DEFAULTS);
	}

	nDummy = 0;
	CsStmCfg.bSaveToFile = (0 != GetPrivateProfileInt(STM_SECTION, _T("SaveToFile"), nDummy, szFilePath));

	nDummy = 0;
	CsStmCfg.bFileFlagNoBuffering = (0 != GetPrivateProfileInt(STM_SECTION, _T("FileFlagNoBuffering"), nDummy, szFilePath));

	nDummy = CsStmCfg.u32TransferTimeout;
	CsStmCfg.u32TransferTimeout = GetPrivateProfileInt(STM_SECTION, _T("TimeoutOnTransfer"), nDummy, szFilePath);

	nDummy = CsStmCfg.u32BufferSizeBytes;
	CsStmCfg.u32BufferSizeBytes =  GetPrivateProfileInt(STM_SECTION, _T("BufferSize"), nDummy, szFilePath);

	nDummy = 0;
	CsStmCfg.bErrorHandling = (0 != GetPrivateProfileInt(STM_SECTION, _T("ErrorHandlingMode"), nDummy, szFilePath));
	
	nDummy = 0;
	CsStmCfg.bCascadeResult = (0 != GetPrivateProfileInt(STM_SECTION, _T("CascadeResult"), nDummy, szFilePath));

	nDummy = 0;
	CsStmCfg.u32DelayStartTransfer = GetPrivateProfileInt(STM_SECTION, _T("DelayStartDMA"), nDummy, szFilePath);

	nDummy = 0;
	CsStmCfg.DataPackCfg = GetPrivateProfileInt(STM_SECTION, _T("DataPackMode"), nDummy, szFilePath);

	nDummy = 0;
	CsStmCfg.u32TimeCounter = GetPrivateProfileInt(STM_SECTION, _T("TimeCounter"), nDummy, szFilePath);

	_stprintf(szDefault, _T("%s"), CsStmCfg.strResultFile);
	GetPrivateProfileString(STM_SECTION, _T("DataFile"), szDefault, szString, MAX_PATH, szFilePath);
	_tcscpy(CsStmCfg.strResultFile, szString);

	*pConfig = CsStmCfg;
	return (CS_SUCCESS);
}

/***************************************************************************************************
****************************************************************************************************/

int32 LoadGpuConfiguration(LPCTSTR szIniFile, PCSGPUCONFIG pConfig)
{
	TCHAR	szDefault[MAX_PATH];
	TCHAR	szString[MAX_PATH];
	TCHAR	szFilePath[MAX_PATH];

	CSGPUCONFIG CsGpuCfg;

	// Set defaults in case we can't read the ini file
	CsGpuCfg.u32SkipFactor = DEFAULT_SKIP_FACTOR;
	CsGpuCfg.i32GpuBlocks = 0;
	CsGpuCfg.i32GpuThreads = -1;
	CsGpuCfg.bDoAnalysis = DEFAULT_DO_ANALYSIS;
	CsGpuCfg.bUseGpu = DEFAULT_USE_GPU;
	strcpy(CsGpuCfg.strResultFile, _T(OUT_FILE));


	if (NULL == pConfig)
	{
		return (CS_INVALID_PARAMETER);
	}

	GetFullPathName(szIniFile, MAX_PATH, szFilePath, NULL);

	if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(szFilePath))
	{
		*pConfig = CsGpuCfg;
		return (CS_USING_DEFAULTS);
	}

	if (0 == GetPrivateProfileSection(GPU_SECTION, szString, 100, szFilePath))
	{
		*pConfig = CsGpuCfg;
		return (CS_USING_DEFAULTS);
	}

	int nDefault = CsGpuCfg.bDoAnalysis;
	CsGpuCfg.bDoAnalysis = (0 != GetPrivateProfileInt(GPU_SECTION, _T("DoAnalysis"), nDefault, szFilePath));

	nDefault = CsGpuCfg.bUseGpu;
	CsGpuCfg.bUseGpu = (0 != GetPrivateProfileInt(GPU_SECTION, _T("UseGpu"), nDefault, szFilePath));

	nDefault = CsGpuCfg.u32SkipFactor;
	CsGpuCfg.u32SkipFactor = GetPrivateProfileInt(GPU_SECTION, _T("SkipFactor"), nDefault, szFilePath);

	nDefault = CsGpuCfg.i32GpuBlocks;
	CsGpuCfg.i32GpuBlocks = GetPrivateProfileInt(GPU_SECTION, _T("GPUBlocks"), nDefault, szFilePath);

	nDefault = CsGpuCfg.i32GpuThreads;
	CsGpuCfg.i32GpuThreads = GetPrivateProfileInt(GPU_SECTION, _T("GPUThreads"), nDefault, szFilePath);

	_stprintf(szDefault, _T("%s"), CsGpuCfg.strResultFile);
	GetPrivateProfileString(GPU_SECTION, _T("ResultsFile"), szDefault, szString, MAX_PATH, szFilePath);
	_tcscpy(CsGpuCfg.strResultFile, szString);
	_tcscat(CsGpuCfg.strResultFile, ".txt");

	*pConfig = CsGpuCfg;
	return (CS_SUCCESS);
}

/***************************************************************************************************
****************************************************************************************************/

cudaError_t InitializeCudaDevice(int32 nDevice, int32* i32MaxBlocks, int32* i32MaxThreadsPerBlock, BOOL* bPinGenericMemory)
{
	// Checking for compute capabilities
	struct cudaDeviceProp deviceProp;  // we need the struct keyword for C compilatio
	cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, nDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "\ncudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	if (0 == deviceProp.canMapHostMemory)
	{
		*bPinGenericMemory = FALSE;
	}
	*i32MaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

	*i32MaxBlocks = deviceProp.maxGridSize[0];

	// Choose which GPU to run on, change this on a multi-GPU system.

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "\ncudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return cudaStatus;
	}
	return cudaSuccess;
}

/***************************************************************************************************
****************************************************************************************************/

DWORD WINAPI CardStreamThread( void *CardIndex )
{
	uInt16				nCardIndex = *((uInt16 *) CardIndex);
	void*				pBuffer1 = NULL;
	void*				pBuffer2 = NULL;
	void*				pBuffer3 = NULL;

	void*				h_buffer1 = NULL;
	void*				h_buffer2 = NULL;
	void*				h_buffer3 = NULL;
	void*				d_buffer1 = NULL;
	void*				d_buffer2 = NULL;
	void*				d_buffer3 = NULL;
	void*				d_buffer = NULL;

	void*				pCurrentBuffer = NULL;
	void*				pWorkBuffer = NULL;
	int* dev_a, * d_accTemp, * d_accTemp2;

	uInt32				u32TransferSizeSamples = 0;
	uInt32				u32SectorSize = 256;
	uInt32				u32DmaBoundary = 16;
	uInt32				u32WriteSize;
	int32				i32Status;

	BOOL				bDone = FALSE;
	uInt32				u32LoopCount = 0;
	uInt32				u32ErrorFlag = 0;
	HANDLE				WaitEvents[2];
	DWORD				dwWaitStatus;
	DWORD				dwRetCode = 0;
	DWORD				dwBytesSave = 0;
	HANDLE				hFile = NULL;
	BOOL				bWriteSuccess = TRUE;
	DWORD				dwFileFlag = g_StreamConfig.bFileFlagNoBuffering ? FILE_FLAG_NO_BUFFERING : 0;
	TCHAR				szSaveFileName[MAX_PATH];
	uInt32				u32ActualLength = 0;
	uInt8				u8EndOfData = 0;
	BOOL				bStreamCompletedSuccess = FALSE;
	cudaError_t			cudaStatus = 0;





	LARGE_INTEGER temp, start_time = { 0 }, end_time = { 0 };
	QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
	double freq = ((double)temp.QuadPart) / 1000.0;

	sprintf_s( szSaveFileName, sizeof (szSaveFileName), "%s_%d.dat", g_StreamConfig.strResultFile, nCardIndex );

	if ( g_StreamConfig.bSaveToFile )
	{
		//If there is an header, the file exist and we must keep the file and don't overwrite it
		hFile = CreateFile( szSaveFileName, GENERIC_READ|GENERIC_WRITE, 0, NULL, CREATE_NEW, dwFileFlag, NULL );
		if ( INVALID_HANDLE_VALUE == hFile )
		{
			_ftprintf (stderr, _T("\nUnable to create data file.\n"));
			ExitThread(1);
		}
	}

/*
	We need to allocate a buffer for transferring the data. Buffer is allocated as void with
	a size of length * number of channels * sample size. All channels in the mode are transferred
	within the same buffer, so the mode tells us the number of channels.  Currently, TAIL_ADJUST 
	samples are placed at the end of each segment. These samples contain timestamp information for the 
	segemnt.  The buffer must be allocated by a call to CsStmAllocateBuffer.  This routine will
	allocate a buffer suitable for streaming.  In this program we're allocating 2 streaming buffers
	so we can transfer to one while doing analysis on the other.
*/

	u32SectorSize = GetSectorSize();
	if ( g_StreamConfig.bFileFlagNoBuffering )
	{
		// If bFileFlagNoBuffering is set, the buffer size should be multiple of the sector size of the Hard Disk Drive.
		// Most of HDDs have the sector size equal 512 or 1024.
		// Round up the buffer size into the sector size boundary
		u32DmaBoundary = u32SectorSize;
	}

	// Round up the DMA buffer size to DMA boundary (required by the Streaming data transfer)
	if ( g_StreamConfig.u32BufferSizeBytes % u32DmaBoundary )
		g_StreamConfig.u32BufferSizeBytes += (u32DmaBoundary - g_StreamConfig.u32BufferSizeBytes % u32DmaBoundary);

	_ftprintf (stderr, _T("\n(Actual buffer size used for data streaming = %u Bytes)\n"), g_StreamConfig.u32BufferSizeBytes );

	i32Status = CsStmAllocateBuffer(g_hSystem, nCardIndex, g_StreamConfig.u32BufferSizeBytes, &pBuffer1);
	if (CS_FAILED(i32Status))
	{
		_ftprintf (stderr, _T("\nUnable to allocate memory for stream buffer 1.\n"));
		CloseHandle(hFile);
		DeleteFile(szSaveFileName);
		ExitThread(1);
	}

	//i32Status = CsStmAllocateBuffer(g_hSystem, nCardIndex, g_StreamConfig.u32BufferSizeBytes, &pBuffer2);
	//if (CS_FAILED(i32Status))
	//{
	//	_ftprintf(stderr, _T("\nUnable to allocate memory for stream buffer 2.\n"));
	//	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
	//	CloseHandle(hFile);
	//	DeleteFile(szSaveFileName);
	//	ExitThread(1);
	//}

	//i32Status = CsStmAllocateBuffer(g_hSystem, nCardIndex, g_StreamConfig.u32BufferSizeBytes, &pBuffer3);
	//if (CS_FAILED(i32Status))
	//{
	//	_ftprintf(stderr, _T("\nUnable to allocate memory for stream buffer 3.\n"));
	//	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
	//	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
	//	CloseHandle(hFile);
	//	DeleteFile(szSaveFileName);
	//	ExitThread(1);
	//}

	if (g_GpuConfig.bUseGpu)
	{
		h_buffer1 = (unsigned char*)ALIGN_UP(pBuffer1, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer1, (size_t)g_StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostRegister failed! Error code %d\n", cudaStatus); 
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
			return cudaStatus;
		}
	/*	h_buffer2 = (unsigned char*)ALIGN_UP(pBuffer2, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer2, (size_t)g_StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostRegister failed! Error code %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
			cudaHostUnregister(h_buffer1);
			return cudaStatus;
		}
		h_buffer3 = (unsigned char*)ALIGN_UP(pBuffer3, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer3, (size_t)g_StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostRegister failed! Error code %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer3);
			cudaHostUnregister(h_buffer1);
			return cudaStatus;
		}*/
		cudaStatus = cudaHostGetDevicePointer((void**)&d_buffer1, (void *)h_buffer1, 0);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostGetDevicePointer failed!  Error code %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
			cudaHostUnregister(h_buffer1);
			cudaHostUnregister(h_buffer2);
			return cudaStatus;
		}
		//cudaStatus = cudaHostGetDevicePointer((void**)&d_buffer2, (void*)h_buffer2, 0);
		//if (cudaStatus != cudaSuccess)
		//{
		//	fprintf(stderr, "cudaHostGetDevicePointer failed!  Error code %d\n", cudaStatus);
		//	CsFreeSystem(g_hSystem);
		//	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
		//	cudaHostUnregister(h_buffer1);
		//	cudaHostUnregister(h_buffer2);
		//	return cudaStatus;
		//}
		//cudaStatus = cudaHostGetDevicePointer((void**)&d_buffer3, (void*)h_buffer3, 0);
		//if (cudaStatus != cudaSuccess)
		//{
		//	fprintf(stderr, "cudaHostGetDevicePointer failed!  Error code %d\n", cudaStatus);
		//	CsFreeSystem(g_hSystem);
		//	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer3);
		//	cudaHostUnregister(h_buffer1);
		//	cudaHostUnregister(h_buffer2);
		//	cudaHostUnregister(h_buffer3);
		//	return cudaStatus;
		//}
	}



	// So far so good ...
	// Let the main thread know that this thread is ready for stream
	SetEvent( g_hThreadReadyForStream );

	// Wait for the start acquisition event from the main thread
	WaitEvents[0] = g_hStreamStarted;
	WaitEvents[1] = g_hStreamAbort;
	dwWaitStatus = WaitForMultipleObjects( 2, WaitEvents, FALSE, INFINITE );

	if ( (WAIT_OBJECT_0 + 1) == dwWaitStatus )
	{
		// Aborted from user or error
		CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
		CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
		CloseHandle(hFile);
		if (g_GpuConfig.bUseGpu)
		{
			cudaHostUnregister(h_buffer1);
			cudaHostUnregister(h_buffer2);
			cudaHostUnregister(h_buffer3);
		}
		DeleteFile(szSaveFileName);
		ExitThread(1);
	}

	// Convert the transfer size to BYTEs or WORDs depending on the card.
	u32TransferSizeSamples = g_StreamConfig.u32BufferSizeBytes/g_CsSysInfo.u32SampleSize;

	int* h_odata = (int*)malloc(1 * sizeof(int));
	short* h_dev_a = (short*)malloc(u32TransferSizeSamples * sizeof(short));
	short* h_dev_a2 = (short*)malloc(u32TransferSizeSamples * sizeof(short));
	cudaStatus = cudaMalloc((int**)&dev_a, u32TransferSizeSamples / 48 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp, 1 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp2, 1 * sizeof(int));
	FILE* fptr;
	if (g_StreamConfig.bCascadeResult == 0) fptr = fopen("Analysis.txt", "w");
	if (g_StreamConfig.bCascadeResult == 1) fptr = fopen("Analysis.txt", "a");
	fprintf(fptr, "//////\nBuffer size (Samples)\n%d\nSampling Rate (Hz)\n%d\n///\n", u32TransferSizeSamples, g_CsAcqCfg.i64SampleRate);
	fclose(fptr);
	
	int timer = 0;

	// Steam acqusition has started.
	// loop until either we've done the number of segments we want, or
	// the ESC key was pressed to abort. While we loop, we transfer data into
	// pCurrentBuffer and save pWorkBuffer to hard disk

	clock_t start_Time, current_time;
	double elapsed_time;

	while( ! ( bDone || bStreamCompletedSuccess) )
	{
		// Check if user has aborted or an error has occured
		if ( WAIT_OBJECT_0 == WaitForSingleObject( g_hStreamAbort, 0 ) )
			break;
		if ( WAIT_OBJECT_0 == WaitForSingleObject( g_hStreamError, 0 ) )
			break;

		// Determine where new data transfer data will go. We alternate
		// between our 2 streaming buffers. d_buffer is the pointer to the
		// buffer on the GPU

		/*if (u32LoopCount & 1)
		{
			pCurrentBuffer = pBuffer2;
			if (g_GpuConfig.bUseGpu)
			{
				d_buffer = d_buffer2;
			}
		}
		else
		{
			pCurrentBuffer = pBuffer1;
			if (g_GpuConfig.bUseGpu)
			{
				d_buffer = d_buffer1;
			}
		}*/
		if (g_GpuConfig.bDoAnalysis)
		{
			QueryPerformanceCounter((LARGE_INTEGER *)&start_time);
		}

		

		i32Status = CsStmTransferToBuffer(g_hSystem, nCardIndex, pBuffer1, u32TransferSizeSamples);
		i32Status = CsStmGetTransferStatus(g_hSystem, nCardIndex, g_StreamConfig.u32TransferTimeout, &u32ErrorFlag, &u32ActualLength, &u8EndOfData);

		/*i32Status = CsStmTransferToBuffer(g_hSystem, nCardIndex, pBuffer2, u32TransferSizeSamples);
		i32Status = CsStmGetTransferStatus(g_hSystem, nCardIndex, g_StreamConfig.u32TransferTimeout, &u32ErrorFlag, &u32ActualLength, &u8EndOfData);
		i32Status = CsStmTransferToBuffer(g_hSystem, nCardIndex, pBuffer3, u32TransferSizeSamples);
		i32Status = CsStmGetTransferStatus(g_hSystem, nCardIndex, g_StreamConfig.u32TransferTimeout, &u32ErrorFlag, &u32ActualLength, &u8EndOfData);
		*/
		if (timer == 1) start_Time = clock();
		cudaStatus = GPU_Equation_PlusOne(d_buffer1, g_GpuConfig.u32SkipFactor, g_CsAcqCfg.u32SampleSize, u32TransferSizeSamples, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, u32LoopCount, h_odata, h_dev_a, h_dev_a2, dev_a, d_accTemp, d_accTemp2);
		if (timer == 1) {
			current_time = clock();
			elapsed_time = ((double)(current_time - start_Time)) / CLOCKS_PER_SEC * 1000;
			printf("Elapsed Time: %.2f ms\r", elapsed_time);
		}
		//u32LoopCount++;
		//cudaStatus = GPU_Equation_PlusOne(d_buffer2, g_GpuConfig.u32SkipFactor, g_CsAcqCfg.u32SampleSize, u32TransferSizeSamples, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, u32LoopCount, h_odata, h_dev_a, h_dev_a2, dev_a, d_accTemp, d_accTemp2);
		//
		//u32LoopCount++;
		//cudaStatus = GPU_Equation_PlusOne(d_buffer3, g_GpuConfig.u32SkipFactor, g_CsAcqCfg.u32SampleSize, u32TransferSizeSamples, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, u32LoopCount, h_odata, h_dev_a, h_dev_a2, dev_a, d_accTemp, d_accTemp2);


		if (CS_FAILED(i32Status))
		{
			if ( CS_STM_COMPLETED == i32Status )
				bStreamCompletedSuccess = TRUE;
			else
			{
				SetEvent( g_hStreamError );
				DisplayErrorString(i32Status);
			}
			break;
		}
		// do processing here on pWorkBuffer
		if (g_GpuConfig.bDoAnalysis)
		{
			if (g_GpuConfig.bUseGpu)
			{
			
				//cudaStatus = GPU_Equation_PlusOne(d_buffer, g_GpuConfig.u32SkipFactor, g_CsAcqCfg.u32SampleSize, u32TransferSizeSamples, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, u32LoopCount);

				if (cudaStatus != cudaSuccess)
				{
					SetEvent(g_hStreamError);
					fprintf(stderr, "GPU_Equation_PlusOne() failed. Error code %d\n", cudaStatus);
					break;
				}
			}
			else // use CPU
			{
				i32Status = CPU_Equation_PlusOne(pCurrentBuffer, g_CsAcqCfg.u32SampleSize, 0, u32TransferSizeSamples);

				if (CS_FAILED(i32Status))
				{
					SetEvent(g_hStreamError);
					fprintf(stderr, "CPU_Equation_PlusOne() failed.\n");
					break;
				}
			}
		}


		if ( g_StreamConfig.bSaveToFile && NULL != pWorkBuffer )
		{
			// While data transfer of the current buffer is in progress, save the data from pWorkBuffer to hard disk
			dwBytesSave = 0;
			bWriteSuccess = WriteFile( hFile, pWorkBuffer, g_StreamConfig.u32BufferSizeBytes, &dwBytesSave, NULL );
			if ( ! bWriteSuccess || dwBytesSave != g_StreamConfig.u32BufferSizeBytes )
			{
				_ftprintf (stdout, _T("\nWriteFile() error on card %d !!! (GetLastError() = 0x%x\n"), nCardIndex, GetLastError());
				SetEvent( g_hStreamError );
				bDone = TRUE;
			}
		}

		// Wait for the DMA transfer on the current buffer to complete so we can loop back around to start a new one.
		// The calling thread will sleep until the transfer completes
		if ( CS_SUCCEEDED(i32Status) )
		{
			// Calculate the total of data transfered so far for this card
			g_llCardTotalData[nCardIndex-1] += u32ActualLength ;//////////////////////////////////////////////////////////////////////////////////////////////////////////
			bStreamCompletedSuccess = (0 != u8EndOfData);

			if ( 0 != u32ErrorFlag )
			{
				if ( STM_TRANSFER_ERROR_FIFOFULL & u32ErrorFlag )
				{
					// The Fifo full error has occured at the card level which results data lost.
					// This error occurs when the application is not fast enough to transfer data.
					if ( 0 != g_StreamConfig.bErrorHandling )
					{
						// g_StreamConfig.bErrorHandling != 0
						// Stop as soon as we recieve the FIFO full error from the card
						SetEvent( g_hStreamError );
						_ftprintf (stdout, _T("\nFifo full detected on the card %d !!!\n"), nCardIndex);
						bDone = TRUE;
					}
					else
					{
						// g_StreamConfig.bErrorHandling == 0
						// Transfer all valid data into the PC RAM

						// Althought the Fifo full has occured, there is valid data available on the On-board memory.
						// To transfer these data into the PC RAM, we can keep calling CsStmTransferToBuffer() then CsStmGetTransferStatus()
						// until the function CsStmTransferToBuffer() returns the error CS_STM_FIFO_OVERFLOW.
						// The error CS_STM_FIFO_OVERFLOW indicates that all valid data has been transfered to the PC RAM

						// Do nothing here, go backto the loop CsStmTransferToBuffer() CsStmGetTransferStatus()
					}
				}
				if ( u32ErrorFlag & STM_TRANSFER_ERROR_CHANNEL_PROTECTION )
				{
					// Channel protection error as coccrued
					SetEvent( g_hStreamError );
					_ftprintf (stdout, _T("\nChannel Protection Error on Board %d!!!\n"), nCardIndex);
					bDone = TRUE;
				}
			}
		}
		else
		{
			SetEvent( g_hStreamError );
			bDone = TRUE;

			if ( CS_STM_TRANSFER_TIMEOUT == i32Status )
			{			 
				//	Timeout on CsStmGetTransferStatus().
				//	Data transfer has not yet completed. We can repeat calling CsStmGetTransferStatus() until we get the status success (ie data transfer is completed)
				//	In this sample program, we consider the timeout as an error
				_ftprintf (stdout, _T("\nStream transfer timeout on card %d !!!\n"), nCardIndex);
			}
			else // some other error 
			{
				char szErrorString[255];

				CsGetErrorString(i32Status, szErrorString, sizeof(szErrorString));
				_ftprintf (stdout, _T("\n%s on card %d !!!\n"), szErrorString, nCardIndex);
			}
		}
		if (g_GpuConfig.bDoAnalysis)
		{
			QueryPerformanceCounter((LARGE_INTEGER *)&end_time);
			diff_time[nCardIndex - 1] += ((double)end_time.QuadPart - (double)start_time.QuadPart) / freq;
		}
		pWorkBuffer = pCurrentBuffer;

		u32LoopCount++;
	}


	if (g_GpuConfig.bDoAnalysis)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)&start_time);
	}

	if ( bStreamCompletedSuccess && g_StreamConfig.bSaveToFile && NULL != pWorkBuffer )
	{
		u32WriteSize = u32ActualLength*g_CsSysInfo.u32SampleSize;

		//Apply a right padding with the sector size
		if( g_StreamConfig.bFileFlagNoBuffering )
		{
			uInt8 *pBufTmp = pWorkBuffer;
			u32WriteSize = ((u32WriteSize - 1) / u32SectorSize + 1) * u32SectorSize;

			// clear padding bytes
			if (u32WriteSize > u32ActualLength*g_CsSysInfo.u32SampleSize)
				memset(&pBufTmp[u32ActualLength*g_CsSysInfo.u32SampleSize], 0, u32WriteSize -u32ActualLength*g_CsSysInfo.u32SampleSize);
		}

		// Save the data from pWorkBuffer to hard disk
		bWriteSuccess = WriteFile( hFile, pWorkBuffer, u32WriteSize, &dwBytesSave, NULL );
		if ( ! bWriteSuccess || dwBytesSave != u32WriteSize )
		{
			_ftprintf (stdout, _T("\nWriteFile() error on card %d !!! (GetLastError() = 0x%x\n"), nCardIndex, GetLastError());
			SetEvent( g_hStreamError );
		}
	}
	if (g_GpuConfig.bDoAnalysis)
	{
		QueryPerformanceCounter((LARGE_INTEGER *)&end_time);
		diff_time[nCardIndex - 1] += ((double)end_time.QuadPart - (double)start_time.QuadPart) / freq;
	}
	// Close the data file and free all streaming buffers
	if ( g_StreamConfig.bSaveToFile )
		CloseHandle(hFile);
	
	if (g_GpuConfig.bUseGpu)
	{
		cudaHostUnregister(h_buffer1);
		cudaHostUnregister(h_buffer2);
		cudaHostUnregister(h_buffer3);
	}
	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer3);
	cudaFree(dev_a);
	cudaFree(d_accTemp);
	cudaFreeHost(d_accTemp2);
	free(h_dev_a);
	free(h_dev_a2);
	free(h_odata);

	if ( bStreamCompletedSuccess )
	{
		dwRetCode = 0;
	}
	else
	{
		// Stream operation has been aborted by user or errors
		dwRetCode = 1;
	}

	ExitThread(dwRetCode);
}

/***************************************************************************************************
****************************************************************************************************/

// Prepare streaming by deleting all existing data file that have the same file name
BOOL Prepare_Cleanup()
{
	uInt32		n = 0;
	BOOL		bSuccess = TRUE;
	TCHAR		szSaveFileName[MAX_PATH];
	HANDLE		hFile = NULL;

	if ( g_StreamConfig.bSaveToFile )
	{
		for (n = 1; n <= g_CsSysInfo.u32BoardCount; n++)
		{
			sprintf_s( szSaveFileName, sizeof (szSaveFileName), "%s_%d.dat", g_StreamConfig.strResultFile, n );
			// Check if the file exists on the HDD
			hFile = CreateFile( szSaveFileName, GENERIC_READ|GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL );
			if ( INVALID_HANDLE_VALUE != hFile )
			{
				CloseHandle(hFile);
				bSuccess = DeleteFile(szSaveFileName);
				if (!bSuccess)
				{
					_ftprintf (stderr, _T("\nUnable to delete the existing data file (%s). GetLastError() = 0x%x\n"), szSaveFileName, GetLastError());
					break;
				}
			}
		}
	}

	return bSuccess;
}

/***************************************************************************************************
****************************************************************************************************/

void DisplayResults(int stream,
	int gpu,
	uInt32 u32Mode,
	uInt32 u32SegmentCount,
	int64 i64TransferLength,
	uInt32 u32SampleSize,
	uInt32 u32SkipFactor,
	double time,
	char* filename)
{
	char s[26];
	char str[255];
	char szHeader[255];
	BOOL bFileExists;
	BOOL bWriteToFile = TRUE;
	FILE* file;
	SYSTEMTIME lt;
	GetLocalTime(&lt);


	bFileExists = (-1 == _access(filename, 0)) ? FALSE : TRUE;

	if (bFileExists)
	{
		if (-1 == _access(filename, 2) || -1 == _access(filename, 6))
		{
			printf("\nCannot write to %s\n", filename);
			bWriteToFile = FALSE;
		}
	}

	sprintf_s(szHeader, _countof(szHeader), "\n\nDate\t     Time\tStream\t GPU\tChannels  Records   Samples\t\tBytes\tSkip\tTime (ms)\n\n");
	printf("%s", szHeader);


	sprintf_s(s, _countof(s), "%04d-%02d-%02d  %2d:%2d:%02d", lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute, lt.wSecond);
	sprintf_s(str, _countof(str), "%s\t  %d\t  %d\t  %d\t    %d\t    %I64d\t\t %d\t %d\t%.3f\n", s, stream, gpu, u32Mode, u32SegmentCount, i64TransferLength, u32SampleSize, u32SkipFactor, time);

	printf("%s", str);

	if (bWriteToFile)
	{
		file = fopen(filename, "a");
		if (NULL != file)
		{
			if (!bFileExists) // first time so write the header
			{
				fwrite(szHeader, 1, strlen(szHeader), file);
			}
			fwrite(str, 1, strlen(str), file);
			fclose(file);
		}
	}
}

/***************************************************************************************************
****************************************************************************************************/

void VerifyData(void* buffer, int64 size, unsigned int sample_size)
{
	// Can be used to print out the first 10 and last 10 samples before and after processing to verfiy the processing
	printf("\n\n");
	if (1 == sample_size)
	{
		unsigned char* buffer8 = (unsigned char*)buffer;
		for (int i = 0; i < 10; i++)
		{
			printf("%d ", buffer8[i]);
		}
		printf(" - ");
		for (int64 i = (size - 10); i < size; i++)
		{
			printf("%d ", buffer8[i]);
		}
	}
	else
	{
		short* buffer16 = (short*)buffer;
		for (int i = 0; i < 10; i++)
		{
			printf("%d ", buffer16[i]);
		}
		printf(" - ");
		for (int64 i = (size - 10); i < size; i++)
		{
			printf("%d ", buffer16[i]);
		}
	}
	printf("\n");
}
