/**
 * gotta use stdint so we can use the same type in C++/C
 * could make the example a bit more complicated with some
 * #defines to fix but it's simpler this way.
 */
#include <stdint.h>


#ifdef __cplusplus

#include <string>

class Foo
{
public:
   Foo( int32_t a );
   ~Foo();
   std::string toString();
private:
   int32_t k;
};

#endif

#ifdef __cplusplus
extern "C" 
{
#endif

void*  getFoo( int32_t a );
void   destroyFoo( void *foo );
void   printString( void *foo );

#ifdef __cplusplus
}
#endif

