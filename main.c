#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void show(int n,int m,float array[n][m]){
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            printf("%4.4f\t",array[i][j]);
        }
        printf("\n");
    }
}
void make_zero(int n,int m,float array[n][m] ){
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
           (array[i][j])=0;
        }
    }
}
void main(){
    float a[6][6];
    make_zero(6,6,&a);
    show(6,6,&a);
    int c ,r;
    int n=10000,i=0;
    while (i<n)
    {
        r = rand()%6;
        c = rand()%6;
        a[r][c]+=1;
    
    i++;
    }
    printf("\n");
    show(6,6,&a);


    return 0;
}