package hello;

public class ForForUpSide1 {
	public static void main(String[] args) {
		for(int a=1;a<=5;a++) {
			for(int b=5;b>0;b--) {
				if(b<=a) {
					System.out.print("*");
				}else {
					System.out.print(" ");
				}
			}
		System.out.println();
		}
	}
}
