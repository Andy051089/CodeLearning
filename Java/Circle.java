/**
 * 通過鍵盤輸入，求圓柱體的體積
 */
package hello;
import java.util.Scanner;
/**
引用JAVA裡util裡scanner這工具
*/
public class Circle {
	public static void main(String[] args){
		Scanner input = new Scanner(System.in);
		/**
		建一個鍵盤輸入的工具
		 */
		System.out.println("請輸入圓柱體的高:");
		int h = input.nextInt();
		/**
		從鍵盤接收一個整數
		 */
		System.out.println("請輸入圓柱體底面的半徑:");
		int r = input.nextInt();
		double ans = r*r*h*3.14;
		/**
		有小數點不能用int
		*/
		System.out.println("圓柱體的體積="+ans);
		}
}

