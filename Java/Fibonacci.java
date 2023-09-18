package hello;

public class Fibonacci {
	public static void main(String[] args) {
		int n=1;
		int n1=1;
		int t=0;
		System.out.println(n);
		System.out.println(n1);
		for(int i=0;i<8;i++) {
			t=n+n1;
			System.out.println(t);
			n=n1;
			n1=t;
		}
	}
}
