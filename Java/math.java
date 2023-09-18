package hello;

public class math {

	public static void main(String[] args) {
		System.out.println("a=10,b=3");
		int a = 10;
		int b = 3;
		int c = a+b;
		System.out.println("a+b="+c);
		int d = a-b;
		System.out.println("a-b="+d);
		int e = a*b;
		System.out.println("a*b="+e);
		int f = a/b;
		System.out.println("a/b="+f);
		int g = a%b;
		System.out.println("a%b="+g);
		/**
		   % 取餘
		 */
		int h = b++;
		/**
		 * 後自增，先把b給a之後在加
		 */
		System.out.println("h="+h);
		System.out.println("b="+b);
		int i = ++b;
		/**
		 * 前自增，先把b給a之後在加
		 */
		System.out.println("i="+i);
		System.out.println("b="+b);
	}
}
