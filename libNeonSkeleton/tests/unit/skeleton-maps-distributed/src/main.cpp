#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include <map>

int main_argc;
char** main_argv;

int main(int argc, char** argv)
{
	main_argc = argc;
	main_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
