package hello;

public class TureFalse {
	public static void main(String[] args) {
		boolean a = false;
		boolean b = false;
		System.out.println(a&b);
		/**
		 * 與運算，分別計算兩邊表達式的結果再作&運算，只兩個都為TURE，結果才會是TURE，否則為FALSE
		 */
		System.out.println(a|b);
		/**
		 * 或運算，分別計算表達式兩邊結果再作|運算，只要有一個為TURE，結果就為TURE，兩邊FALSE，才為FALSE
		 */
		System.out.println(a^b);
		/**
		 * 異或運算，兩邊相同為FALSE，不同為TURE
		 */
		System.out.println(a&&b);
		System.out.println(a||b);
		System.out.println(!b);
		
		int c = 1;
		int d = 2;
		c = c^d;
		d = c^d;
		c = c^d;	
		System.out.println("c="+c+","+"d="+d);
		/**
		 * 使用三次異或運算，是最快更換兩數方式
		 */
	}
}
