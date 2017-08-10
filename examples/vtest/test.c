#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void memory_leak( void)
{
    char    *p_leak;
    int      ndx;
    
    p_leak  = malloc(  10);
    
    for ( ndx = 0; ndx < 10; ndx++)
    {
        p_leak[ndx] = 'a';
    }    
    for ( ndx = 0; ndx < 10; ndx++)
    {
        printf( "%c", p_leak[ndx]);
    }    
}

int main( void)
{
    memory_leak();
}

