int main(int argc, char **argv)
{
	TAU_PROFILE("main()", " ", TAU_DEFAULT);
	TAU_PROFILE_SET_NODE(0);

	TAU_TRACK_MEMORY();

	sleep(12);

	int *x = new int[5*1024*1024];

	sleep(12);

	return 0;
}



