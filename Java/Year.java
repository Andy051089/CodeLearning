/**
 * 一個年份是閏年
 */
package hello;

public class Year {
	public static void main(String[] args) {
	boolean ans = runYear(2017);
		if(ans==true) {
			System.out.println("閏年");
		}else {
			System.out.println("平年");
		}
	}
	public static boolean runYear(int year) {
		if((year%4==0 && year%100!=0) || year%400==0) {
			return true;
		}else {
			return false;
		}
	}
}
