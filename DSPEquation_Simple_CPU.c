

int CPU_Equation_PlusOne(void* buffer, unsigned long sample_size, __int64 start, __int64 length)
{
	__int64 i;

	if (!buffer || length < 0)
	{
		return -1;
	}

	if (1 == sample_size)
	{
		unsigned char* buffer8 = (unsigned char*)buffer;

		for (i = start; i < start + length; i++)
		{
			buffer8[i] = buffer8[i] + 1;
		}
	}
	else
	{
		short* buffer16 = (short*)buffer;

		for (i = start; i < start + length; i++)
		{
			buffer16[i] = buffer16[i] + 1;
		}
	}
	return 1;
}