package hello;

public class ForForTriangle {
	public static void main(String[] args) {
		for(int i=1;i<=5;i++) {
			for(int j=i;j<5;j++) {
				System.out.print(" ");
			}
			for(int a=1;a<=i*2-1;a++) {
				System.out.print("*");
			}
			System.out.println();
		}
	}
}
		