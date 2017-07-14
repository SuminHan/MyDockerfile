#include <stdlib.h>
#include "foo.h"

int main(const int argc, const char **argv)
{
   
   void *temp_foo = getFoo(5);

   printString(temp_foo);

   return( EXIT_SUCCESS );
}
